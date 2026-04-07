import os
import json
import time
import threading
from datetime import datetime
from apscheduler.events import EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from backend.core.schemas import Message, Component, MessageType, SendType
from backend.core.log import get_logger

logger = get_logger("schedule_manager")

class ScheduleManager:
    def __init__(self, gateway):
        self.gateway = gateway
        self.scheduler = BackgroundScheduler()
        
        # 目录与文件初始化
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../workspace/schedules'))
        self.pending_file = os.path.join(self.base_dir, 'pending_tasks.json')
        self.history_dir = os.path.join(self.base_dir, 'history')
        
        os.makedirs(self.history_dir, exist_ok=True)
        if not os.path.exists(self.pending_file):
            with open(self.pending_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
                
        self._last_mtime = 0
        self.scheduler.add_listener(self._on_job_missed, EVENT_JOB_MISSED)
        self.scheduler.start()
        
        # 启动文件轮询监听线程
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_tasks, daemon=True)
        self._poll_thread.start()

    def stop(self):
        self._running = False
        self.scheduler.shutdown()
        
    def _poll_tasks(self):
        """定期检查 JSON 文件是否有更新"""
        while self._running:
            try:
                if os.path.exists(self.pending_file):
                    current_mtime = os.path.getmtime(self.pending_file)
                    if current_mtime > self._last_mtime:
                        self._sync_jobs()
                        self._last_mtime = current_mtime
            except Exception as e:
                logger.error(f"Schedule file polling error: {e}")
            time.sleep(2)
            
    def _sync_jobs(self):
        """对比 JSON 与内存调度器，进行热重载"""
        try:
            with open(self.pending_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
                
            task_ids = {t['task_id']: t for t in tasks}
            
            # 添加或更新
            for task_id, task in task_ids.items():
                if not self.scheduler.get_job(task_id):
                    self._add_job_to_scheduler(task)
                    
            # 移除已删除的
            for job in self.scheduler.get_jobs():
                if job.id not in task_ids:
                    self.scheduler.remove_job(job.id)
        except Exception as e:
            logger.error(f"Failed to sync schedule jobs: {e}")

    def _on_job_missed(self, event):
        """APScheduler 运行时错失任务的事件回调"""
        task_id = event.job_id
        try:
            if os.path.exists(self.pending_file):
                with open(self.pending_file, 'r', encoding='utf-8') as f:
                    tasks = json.load(f)
                for t in tasks:
                    if t['task_id'] == task_id:
                        self._handle_expired_task(t)
                        break
        except Exception as e:
            logger.error(f"Error handling missed job event: {e}")

    def _handle_expired_task(self, task: dict):
        """处理已过期的任务：计算延误时间并向用户发送兜底确认"""
        run_date_str = task.get('trigger_args', {}).get('run_date')
        action_type = task.get('action_type')
        content = task.get('content', '')
        
        try:
            # 解析目标时间
            try:
                run_date = datetime.fromisoformat(run_date_str)
            except ValueError:
                run_date = datetime.strptime(run_date_str, '%Y-%m-%d %H:%M:%S')
                
            delay = datetime.now() - run_date
            total_seconds = int(delay.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            
            delay_str = ""
            if hours > 0: delay_str += f"{hours}小时"
            if minutes > 0: delay_str += f"{minutes}分钟"
            if not delay_str: delay_str = "不到1分钟"

            if action_type == 'send_message':
                msg = Message(
                    sender=Component.SYSTEM,
                    send_type=SendType.USER,
                    message_type=MessageType.SCHEDULE,
                    content=f"【延误提醒】\n该提醒本应在 {run_date_str} 发送，已延误 {delay_str}。\n内容：{content}"
                )
                self.gateway.handle(msg)
                
            elif action_type == 'create_task':
                msg = Message(
                    sender=Component.SYSTEM,
                    send_type=SendType.USER,
                    message_type=MessageType.SCHEDULE,
                    content=f"【过期任务确认】\n有一个计划在 {run_date_str} 执行的任务，因故延误了 {delay_str}。\n内容：{content}\n请问现在是否还需要执行此任务？"
                )
                self.gateway.handle(msg)

            # 归档该任务，标记为过期已通知
            self._check_and_archive(task['task_id'], force_archive=True, status="expired_notified")
            
        except Exception as e:
            logger.error(f"Error handling expired task {task.get('task_id')}: {e}")
                
    def _add_job_to_scheduler(self, task: dict):
        trigger_type = task.get('trigger_type')
        trigger_args = task.get('trigger_args', {})
        
        try:
            trigger_instance = None
            kwargs_trigger = {}
            
            if trigger_type == 'date':
                run_date_str = trigger_args.get('run_date')
                
                # 装载前的主动过期拦截
                if run_date_str:
                    try:
                        run_date = datetime.fromisoformat(run_date_str)
                    except ValueError:
                        run_date = datetime.strptime(run_date_str, '%Y-%m-%d %H:%M:%S')
                        
                    if run_date < datetime.now():
                        self._handle_expired_task(task)
                        return
                trigger_instance = 'date'
                kwargs_trigger = {'run_date': trigger_args.get('run_date')}
            elif trigger_type == 'cron':
                cron_expr = trigger_args.get('cron_expression')
                # 兼容 Linux 风格的 5 位 cron 表达式 (分 时 日 月 周)
                trigger_instance = CronTrigger.from_crontab(cron_expr)

            if trigger_instance is None:
                return

            if isinstance(trigger_instance, str):
                self.scheduler.add_job(
                    func=self._execute_task,
                    trigger=trigger_instance,
                    kwargs={'task': task},
                    id=task['task_id'],
                    **kwargs_trigger
                )
            else:
                self.scheduler.add_job(
                    func=self._execute_task,
                    trigger=trigger_instance,
                    kwargs={'task': task},
                    id=task['task_id']
                )
            logger.info(f"Added schedule job: {task['task_id']}")
        except Exception as e:
            logger.error(f"Error adding trigger {trigger_type} for task {task['task_id']}: {e}")

    def _execute_task(self, task: dict):
        """到期执行：根据类型向 Gateway 抛出对应的 Message"""
        action_type = task.get('action_type')
        content = task.get('content', '')
        
        try:
            if action_type == 'send_message':
                msg = Message(
                    sender=Component.SYSTEM,
                    send_type=SendType.USER,
                    message_type=MessageType.SCHEDULE,
                    content=f"【定时提醒】\n{content}"
                )
                self.gateway.handle(msg)
                
            elif action_type == 'create_task':
                dispatch_content = f"【系统定时派发任务】\n{content}"
                    
                msg = Message(
                    sender=Component.BUTLER,
                    send_type=SendType.DOWNWARD,
                    message_type=MessageType.SCHEDULE,
                    content=dispatch_content,
                    data={
                        "permission_type": "smart"
                    }
                )
                self.gateway.handle(msg)
        except Exception as e:
            logger.error(f"Error executing schedule task {task['task_id']}: {e}")
            
        # 检查是否为一次性任务，如果是则归档
        self._check_and_archive(task['task_id'])

    def _check_and_archive(self, task_id: str, force_archive: bool = False, status: str = "executed"):
        """一次性任务结束后，移入历史归档"""
        job = self.scheduler.get_job(task_id)
        # 如果强制归档，或者调度器中已没有后续运行计划
        if force_archive or job is None or job.next_run_time is None:
            try:
                with open(self.pending_file, 'r', encoding='utf-8') as f:
                    tasks = json.load(f)
                
                target_task = None
                new_tasks = []
                for t in tasks:
                    if t['task_id'] == task_id:
                        target_task = t
                    else:
                        new_tasks.append(t)
                        
                if target_task:
                    # 覆写 pending
                    with open(self.pending_file, 'w', encoding='utf-8') as f:
                        json.dump(new_tasks, f, ensure_ascii=False, indent=2)
                        
                    # 追加写入 history
                    today = datetime.now().strftime('%Y-%m-%d')
                    history_file = os.path.join(self.history_dir, f"{today}.json")
                    
                    history = []
                    if os.path.exists(history_file):
                        with open(history_file, 'r', encoding='utf-8') as f:
                            history = json.load(f)
                            
                    target_task['executed_at'] = datetime.now().isoformat()
                    target_task['status'] = status  # 使用传入的状态
                    history.append(target_task)
                    
                    with open(history_file, 'w', encoding='utf-8') as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Archive schedule error: {e}")