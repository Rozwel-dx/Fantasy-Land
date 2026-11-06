# buffer_size太小，可能会导致丢trace，可以根据需要设置，例如409600
echo 409600 > /sys/kernel/debug/tracing/buffer_size_kb
echo nop > /sys/kernel/debug/tracing/current_tracer
echo 0 > /sys/kernel/debug/tracing/tracing_on
echo 0 > /sys/kernel/debug/tracing/trace
echo 1 > /sys/kernel/debug/tracing/events/sched/sched_migrate_task/enable
echo 1 > /sys/kernel/debug/tracing/events/sched/sched_wakeup/enable
echo 1 > /sys/kernel/debug/tracing/events/sched/sched_waking/enable
echo 1 > /sys/kernel/debug/tracing/events/sched/sched_switch/enable
#中断
echo 1 > /sys/kernel/debug/tracing/events/irq/irq_handler_entry/enable
echo 1 > /sys/kernel/debug/tracing/events/irq/irq_handler_exit/enable
echo 1 > /sys/kernel/debug/tracing/events/irq/softirq_entry/enable
echo 1 > /sys/kernel/debug/tracing/events/irq/softirq_exit/enable
echo 1 > /sys/kernel/debug/tracing/events/irq/softirq_raise/enable
echo 1 > /sys/kernel/debug/tracing/events/sched/sched_wakeup_new/enable
echo 1 > /sys/kernel/debug/tracing/events/task/task_newtask/enable
echo 1 > /sys/kernel/debug/tracing/events/power/cpu_idle/enable

# 新增
# 系统软中断跟踪
echo 1 > /sys/kernel/debug/tracing/events/irq/softirq_entry/enable
echo 1 > /sys/kernel/debug/tracing/events/irq/softirq_exit/enable
echo 1 > /sys/kernel/debug/tracing/events/irq/softirq_raise/enable

# 系统硬中断跟踪
echo 1 >  /sys/kernel/debug/tracing/events/irq/irq_handler_entry/enable
echo 1 >  /sys/kernel/debug/tracing/events/irq/irq_handler_exit/enable

#系统软中断堆栈跟踪
# echo 'traceoff' > /sys/kernel/debug/tracing/events/irq/softirq_entry/trigger
echo 'stacktrace' > /sys/kernel/debug/tracing/events/irq/softirq_entry/trigger

# 系统所有调用跟踪
echo 1 > /sys/kernel/debug/tracing/events/raw_syscalls/sys_enter/enable
echo 1 > /sys/kernel/debug/tracing/events/raw_syscalls/sys_exit/enable

# 针对特定系统调用(Futex)的跟踪
echo 1 > /sys/kernel/debug/tracing/events/syscalls/sys_enter_futex/enable
echo 1 > /sys/kernel/debug/tracing/events/syscalls/sys_exit_futex/enable

# 任务调度跟踪
echo 1 > /sys/kernel/debug/tracing/events/sched/sched_switch/enable
echo 1 > /sys/kernel/debug/tracing/events/sched/sched_wakeup/enable
echo 1 > /sys/kernel/debug/tracing/events/sched/sched_waking/enable
echo 1 > /sys/kernel/debug/tracing/events/sched/sched_migrate_task/enable
echo 1 > /sys/kernel/debug/tracing/events/sched/sched_wakeup_new/enable

# 如果多次采集，报file exist之类的错，可以在每一个echo stacktrace之前，echo 'traceoff'
# echo stacktrace > /sys/kernel/debug/tracing/events/irq/softirq_entry/trigger
# echo stacktrace > /sys/kernel/debug/tracing/events/irq/softirq_exit/trigger
# echo stacktrace > /sys/kernel/debug/tracing/events/irq/irq_handler_entry/trigger
# echo stacktrace > /sys/kernel/debug/tracing/events/sched/sched_migrate_task/trigger
# CPU mask：从右到左是CPU的低位到高位，每一个f代表4个CPU全部采集（即1111的16进制为f）。
# 不建议采集所有的CPU，10个CPU采集10s数据量大概为数百兆
# echo 000e0000,e0000e00,00e0000e,0000e000,0e0000e0,000e0000,e0000e00,00e0000e,0000e000,0e0000e0 > /sys/kernel/debug/tracing/tracing_cpumask
echo ffffffff,ffffffff,ffffffff,ffffffff,ffffffff,ffffffff > /sys/kernel/debug/tracing/tracing_cpumask
# echo 00040000,40000400,00400004,00004000,04000040,00040000,40000400,00400004,00004000,04000040 > /sys/kernel/debug/tracing/tracing_cpumask

echo "开始时间：$(date "+%Y-%m-%d %H:%M:%S")"
echo 1 > /sys/kernel/debug/tracing/tracing_on
sleep 3
echo 0 > /sys/kernel/debug/tracing/tracing_on
echo "结束时间：$(date "+%Y-%m-%d %H:%M:%S")"
echo 0 > /sys/kernel/debug/tracing/events/sched/sched_migrate_task/enable
echo 0 > /sys/kernel/debug/tracing/events/sched/sched_wakeup/enable
echo 0 > /sys/kernel/debug/tracing/events/sched/sched_waking/enable
echo 0 > /sys/kernel/debug/tracing/events/sched/sched_switch/enable
echo 0 > /sys/kernel/debug/tracing/events/irq/irq_handler_entry/enable
echo 0 > /sys/kernel/debug/tracing/events/irq/irq_handler_exit/enable
echo 0 > /sys/kernel/debug/tracing/events/irq/softirq_entry/enable
echo 0 > /sys/kernel/debug/tracing/events/irq/softirq_exit/enable
echo 0 > /sys/kernel/debug/tracing/events/irq/softirq_raise/enable
echo 0 > /sys/kernel/debug/tracing/events/sched/sched_wakeup_new/enable
echo 0 > /sys/kernel/debug/tracing/events/task/task_newtask/enable
echo 0 > /sys/kernel/debug/tracing/events/power/cpu_idle/enable

# 新增
echo 0 > /sys/kernel/debug/tracing/events/irq/softirq_entry/enable
echo 0 > /sys/kernel/debug/tracing/events/irq/softirq_exit/enable
echo 0 > /sys/kernel/debug/tracing/events/irq/softirq_raise/enable

echo 0 > /sys/kernel/debug/tracing/events/irq/irq_handler_entry/enable
echo 0 > /sys/kernel/debug/tracing/events/irq/irq_handler_exit/enable

echo 'traceoff' > /sys/kernel/debug/tracing/events/irq/softirq_entry/trigger

echo 0 > /sys/kernel/debug/tracing/events/raw_syscalls/sys_enter/enable
echo 0 > /sys/kernel/debug/tracing/events/raw_syscalls/sys_exit/enable

echo 0 > /sys/kernel/debug/tracing/events/syscalls/sys_enter_futex/enable
echo 0 > /sys/kernel/debug/tracing/events/syscalls/sys_exit_futex/enable

echo 0 > /sys/kernel/debug/tracing/events/sched/sched_switch/enable
echo 0 > /sys/kernel/debug/tracing/events/sched/sched_wakeup/enable
echo 0 > /sys/kernel/debug/tracing/events/sched/sched_waking/enable
echo 0 > /sys/kernel/debug/tracing/events/sched/sched_migrate_task/enable
echo 0 > /sys/kernel/debug/tracing/events/sched/sched_wakeup_new/enable


# echo 'traceoff' > /sys/kernel/debug/tracing/events/irq/softirq_entry/trigger
# echo 'traceoff' > /sys/kernel/debug/tracing/events/irq/softirq_exit/trigger
# echo 'traceoff' > /sys/kernel/debug/tracing/events/irq/irq_handler_entry/trigger
cat /sys/kernel/debug/tracing/trace > ./cpu_trace_all
# 清空 trace
echo '!traceoff'  > /sys/kernel/debug/tracing/events/irq/softirq_entry/trigger
echo '!stacktrace'  > /sys/kernel/debug/tracing/events/irq/softirq_entry/trigger
echo > /sys/kernel/debug/tracing/trace
