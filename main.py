from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import AGRAWALGenerator
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.meta import OzaBaggingADWINClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import FileStream

from cdda import CDDA

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os

# ----------------
# Constants
# ----------------

classifiers = {'CDDA' : 'CDDA' , 'HAT' : 'HAT', 'OBA' : 'OBA', 'SAMKNN' : 'SAMKNN'}

# Parameters
N = 100000  # Number of Samples
numberOfSources = 3
batchSize = 200
# streamName = CDDA().streamNames['POKR']
# classifierName = classifiers['HAT']

cdda = CDDA(numberOfSamples=N, windowSize=batchSize, numberOfSources=numberOfSources)

allResultPath = 'evals/all_results.csv'

sw = True
for classifierName in [classifiers['SAMKNN']]:
    for streamName in CDDA.streamNames.keys():
# for classifierName in [classifiers['OBA']]:
#     for streamName in [CDDA.streamNames['ELEC'], CDDA.streamNames['POKR']]:
    # for streamName in ['SEA_A', 'SEA_G']:
        stream = cdda.getStream(streamName)
        if classifierName == classifiers['CDDA']:
            cdda.evaluate(stream)
            # cdda.plotStream(stream)
        else:
            if classifierName == classifiers['HAT']:
                classifier = HoeffdingAdaptiveTreeClassifier()
            elif classifierName == classifiers['OBA']:
                classifier = OzaBaggingADWINClassifier()
            elif classifierName == classifiers['SAMKNN']:
                classifier = SAMKNNClassifier()

            evalPath = 'evals/' + classifierName + '_' + streamName + '_' + str(N) + '_eval.csv'
            evaluator = EvaluatePrequential(show_plot=False, pretrain_size=batchSize, max_samples=stream.n_samples,
                                            batch_size=batchSize,
                                            output_file= evalPath,
                                            metrics=['accuracy', 'kappa', 'running_time', 'model_size'])

            evaluator.evaluate(stream=stream, model=classifier)
            metrics = cdda.getEvaluationMetrics(evalPath)

            df = {}
            df['classifier_name'] = classifierName
            df['stream_name'] = streamName
            df['number_of_samples'] = N
            df[cdda.metricNames['ACC_MEAN']] = metrics[cdda.metricNames['ACC_MEAN']]
            df[cdda.metricNames['ACC_STD']] = metrics[cdda.metricNames['ACC_STD']]
            df[cdda.metricNames['KAPPA_MEAN']] = metrics[cdda.metricNames['KAPPA_MEAN']]
            df[cdda.metricNames['KAPPA_STD']] = metrics[cdda.metricNames['KAPPA_STD']]
            df[cdda.metricNames['EXEC_TIME']] = metrics[cdda.metricNames['EXEC_TIME']]
            df[cdda.metricNames['MODEL_SIZE']] = metrics[cdda.metricNames['MODEL_SIZE']]

            print(df)
            df = pd.DataFrame.from_records([df])

            header = False
            if sw:
                header = True
                sw = False
            df.to_csv(allResultPath, mode='a', header=header)


            print('Accuracy : %.2f +- %.2f' % (metrics[cdda.metricNames['ACC_MEAN']], metrics[cdda.metricNames['ACC_STD']]))
            print('Kappa : %.2f +- %.2f' % (metrics[cdda.metricNames['KAPPA_MEAN']], metrics[cdda.metricNames['KAPPA_STD']]))
            print('Execution Time : %.2f' % metrics[cdda.metricNames['EXEC_TIME']])
            print('Model Size : %.2f' % metrics[cdda.metricNames['MODEL_SIZE']])


print(pd.read_csv(allResultPath))