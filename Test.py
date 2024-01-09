import uuid

from algorithm.metric.train import onlineTrain
from entity.TrainTask import TrainTask
from utils.prometheusUtil import PROMETHEUS


if __name__ == "__main__":
    # metricList = ['cpu_usage_30s', 'memory_rate_to_machine_total']
    # metricMap = PROMETHEUS.getMetrics(metricList, 1704449314, 1704449364)
    # PROMETHEUS.fillMetricsWithPreviousValue(metricMap)

    onlineTrain("my_train", None, None)

    print("end")

