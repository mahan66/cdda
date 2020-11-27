import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sys import getsizeof

from skmultiflow.data import DataStream
from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import SEAGenerator
from skmultiflow.data import LEDGenerator
from skmultiflow.data import LEDGeneratorDrift
from skmultiflow.data import RandomRBFGeneratorDrift
from skmultiflow.data import SineGenerator
from skmultiflow.data import HyperplaneGenerator

from skmultiflow.data import FileStream

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import cohen_kappa_score as kappa_score

from scipy.linalg import sqrtm
from numpy.linalg import inv


class CDDA:
    # Class Parameters
    numberOfSources = None
    numberOfSamples = None
    numberOfWindows = None
    windowSize = None

    stream = None

    noisePercentage = 0.1
    balanceClasses = True

    defaultStreamsPath = 'streams/'

    # Stream Names
    streamNames = {'SEA_A': 'SEA_A', 'SEA_G': 'SEA_G', 'LED_A': 'LED_A', 'LED_G': 'LED_G',
                   'RBF_M': 'RBF_M', 'RBF_F': 'RBF_F', 'Sine_A': 'Sine_A', 'Sine_G': 'Sine_G',
                   'HYPER': 'HYPER', 'ELEC': 'ELEC', 'POKR': 'POKR'
                   }

    metricNames = {'ACC_MEAN' : 'acc_mean', 'ACC_STD' : 'acc_std', 'KAPPA_MEAN' : 'kappa_mean',
                   'KAPPA_STD' : 'kappa_std', 'EXEC_TIME' : 'exec_time', 'MODEL_SIZE' : 'model_size'}
    def __init__(self, numberOfSamples=10000, windowSize=100, numberOfSources=2):
        self.numberOfSources = numberOfSources
        self.numberOfSamples = numberOfSamples
        self.windowSize = windowSize
        self.numberOfWindows = int(self.numberOfSamples / self.windowSize)

    def getStream(self, streamName=streamNames['SEA_A']):
        print(streamName)
        streamFilePath = self.createStream(streamName)
        print(streamFilePath)
        self.stream = FileStream(streamFilePath)
        return self.stream

    def saveAndLoadStreams(self, streamName=streamNames['SEA_A']):
        filePath = 'streams/' + streamName + '-N_' + str(self.numberOfSamples) + '.csv'

        if not os.path.exists(filePath):
            X, y = self.stream.next_sample(self.numberOfSamples)
            df = pd.DataFrame(np.hstack((X, np.array([y]).T)))
            columns = self.stream.feature_names

            targetNames = []
            print(self.stream.n_targets)
            if self.stream.target_names is None:
                for i in range(self.stream.n_targets):
                    targetNames.append('target_' + str(i))
            else:
                targetNames = self.stream.target_names

            for targetName in targetNames:
                columns.append(targetName)
            df.columns = columns
            df.iloc[:, -1] = [int(x) for x in df.iloc[:, -1]]

            df.to_csv(filePath, index=False)

        return filePath

    def createStream(self, streamName=streamNames['SEA_A']):
        streamFilePath = ''
        if streamName == self.streamNames['SEA_A']:
            self.stream = ConceptDriftStream(
                stream=SEAGenerator(classification_function=1, balance_classes=self.balanceClasses),
                drift_stream=SEAGenerator(classification_function=3, balance_classes=self.balanceClasses,
                                          noise_percentage=self.noisePercentage),
                position=int(self.numberOfSamples / 2), width=1)
            streamFilePath = self.saveAndLoadStreams(self.streamNames['SEA_A'])

        elif streamName == self.streamNames['SEA_G']:
            self.stream = ConceptDriftStream(
                stream=SEAGenerator(classification_function=1, balance_classes=self.balanceClasses),
                drift_stream=SEAGenerator(classification_function=2, balance_classes=self.balanceClasses,
                                          noise_percentage=self.noisePercentage),
                position=int(self.numberOfSamples / 2), width=self.windowSize * 100)
            streamFilePath = self.saveAndLoadStreams(self.streamNames['SEA_G'])

        elif streamName == self.streamNames['LED_A']:
            self.stream = ConceptDriftStream(
                stream=LEDGenerator(random_state=1, noise_percentage=0.1, has_noise=True),
                drift_stream=ConceptDriftStream(
                    stream=LEDGeneratorDrift(random_state=1, noise_percentage=0.1, has_noise=True, n_drift_features=4),
                    drift_stream=ConceptDriftStream(
                        stream=LEDGeneratorDrift(random_state=1, noise_percentage=0.1, has_noise=True,
                                                 n_drift_features=3),
                        drift_stream=LEDGeneratorDrift(random_state=1, noise_percentage=0.1, has_noise=True,
                                                       n_drift_features=7),
                        position=int(self.numberOfSamples / 4), width=1),
                    position=int(self.numberOfSamples / 2), width=1),
                position=int(self.numberOfSamples / 2 + self.numberOfSamples / 4), width=1)
            streamFilePath = self.saveAndLoadStreams(self.streamNames['LED_A'])

        elif streamName == self.streamNames['LED_G']:
            self.stream = ConceptDriftStream(
                stream=LEDGenerator(random_state=1, noise_percentage=0.1, has_noise=True),
                drift_stream=ConceptDriftStream(
                    stream=LEDGeneratorDrift(random_state=1, noise_percentage=0.1, has_noise=True, n_drift_features=4),
                    drift_stream=ConceptDriftStream(
                        stream=LEDGeneratorDrift(random_state=1, noise_percentage=0.1, has_noise=True,
                                                 n_drift_features=3),
                        drift_stream=LEDGeneratorDrift(random_state=1, noise_percentage=0.1, has_noise=True,
                                                       n_drift_features=7),
                        position=int(self.numberOfSamples / 4), width=1),
                    position=int(self.numberOfSamples / 2), width=1),
                position=int(self.numberOfSamples / 2 + self.numberOfSamples / 4), width=self.windowSize * 100)
            streamFilePath = self.saveAndLoadStreams(self.streamNames['LED_G'])

        elif streamName == self.streamNames['RBF_M']:
            self.stream = RandomRBFGeneratorDrift(model_random_state=99, sample_random_state=50,
                                                  n_classes=4, n_features=10, n_centroids=50, change_speed=0.0001,
                                                  num_drift_centroids=50)
            streamFilePath = self.saveAndLoadStreams(self.streamNames['RBF_M'])

        elif streamName == self.streamNames['RBF_F']:
            self.stream = RandomRBFGeneratorDrift(model_random_state=99, sample_random_state=50,
                                                  n_classes=4, n_features=10, n_centroids=50, change_speed=0.001,
                                                  num_drift_centroids=50)
            streamFilePath = self.saveAndLoadStreams(self.streamNames['RBF_F'])

        elif streamName == self.streamNames['Sine_A']:
            self.stream = ConceptDriftStream(
                stream=SineGenerator(classification_function=1, balance_classes=self.balanceClasses),
                drift_stream=SineGenerator(classification_function=2, balance_classes=self.balanceClasses),
                position=int(self.numberOfSamples / 2), width=1)
            streamFilePath = self.saveAndLoadStreams(self.streamNames['Sine_A'])

        elif streamName == self.streamNames['Sine_G']:
            self.stream = ConceptDriftStream(
                stream=SineGenerator(classification_function=1, balance_classes=self.balanceClasses),
                drift_stream=SineGenerator(classification_function=2, balance_classes=self.balanceClasses),
                position=int(self.numberOfSamples / 2), width=int(self.numberOfSamples / 100) * 10)
            streamFilePath = self.saveAndLoadStreams(self.streamNames['Sine_G'])

        elif streamName == self.streamNames['HYPER']:
            self.stream = HyperplaneGenerator(mag_change=0.001, noise_percentage=self.noisePercentage,
                                              n_drift_features=4)
            streamFilePath = self.saveAndLoadStreams(self.streamNames['HYPER'])

        elif streamName == self.streamNames['ELEC']:
            convertedElecPath = 'streams/real/converted_electricity.csv'
            if not os.path.exists(convertedElecPath):
                elecPath = 'streams/real/electricity.csv'
                X = pd.read_csv(elecPath)
                X['class'] = X['class'].replace(to_replace=['UP', 'DOWN'], value=[1, 0])
                X.drop(['date'], axis=1, inplace=True)
                print(X.head())
                X.to_csv(convertedElecPath, index=False)

            streamFilePath = convertedElecPath

        elif streamName == self.streamNames['POKR']:
            convertedPokerHandPath = 'streams/real/converted_pokr.csv'
            if not os.path.exists(convertedPokerHandPath):
                pokrPath = 'streams/real/poker-hand-testing.data'
                X = pd.read_csv(pokrPath, names=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'hand'])
                X.to_csv(convertedPokerHandPath, index=False)

            streamFilePath = convertedPokerHandPath

        return streamFilePath

    def evaluate(self, stream: FileStream, model=None):
        X, y = stream.get_all_samples()

        acc = []
        kappa = []
        self.numberOfSamples = X.shape[0]
        self.numberOfWindows = int(self.numberOfSamples / self.windowSize)
        sourcesSampleSize = self.numberOfSources * self.windowSize

        start = time.time()
        memory = 0
        N = int(self.numberOfWindows / self.numberOfSources) - 1
        for i in range(0, N , 1):
            # print(i)
            x_s = X[i * sourcesSampleSize:(i + 1) * sourcesSampleSize, :]
            y_s = np.ravel(y[i * sourcesSampleSize:(i + 1) * sourcesSampleSize, :])

            x_t = X[(i + 1) * sourcesSampleSize:(i + 1) * sourcesSampleSize + self.windowSize, :]
            y_t = np.ravel(y[(i + 1) * sourcesSampleSize:(i + 1) * sourcesSampleSize + self.windowSize, :])

            # print(i, x_s.shape, y_s.shape, x_t.shape, y_t.shape)

            cov_source = (pd.DataFrame(x_s).corr().to_numpy() + np.eye(x_s.shape[1])).clip(min=0.01)
            cov_target = (pd.DataFrame(x_t).corr().to_numpy() + np.eye(x_t.shape[1])).clip(min=0.01)
            # print(cov_source)
            # print(x_s)
            # print(cov_target)
            # print(x_t)
            A_coral = sqrtm(np.matmul(inv(cov_source), cov_target))
            A_coral = np.where(A_coral < 0.001, 0, A_coral)

            x_s = np.matmul(x_s, A_coral)

            # gnb = GaussianNB()
            # gnb = LogisticRegression()
            gnb = svm.SVC()
            # gnb = RandomForestClassifier()

            y_pred = gnb.fit(x_s, y_s).predict(x_t)

            numErr = (y_t != y_pred).sum()
            curAcc = (x_t.shape[0] - numErr) / x_t.shape[0]
            acc.append(curAcc)

            curKappa = kappa_score(y_pred, y_t)
            kappa.append(curKappa)
            print(i, 'ACC: ' + str(curAcc) + '  Kappa: ' + str(curKappa) )

            memory += getsizeof(x_s) + getsizeof(x_t) + getsizeof(y_s) + getsizeof(y_t) + \
                      getsizeof(cov_source) + getsizeof(cov_target) + getsizeof(A_coral) + getsizeof(gnb) + getsizeof(y_pred)
            # print("Number of mislabeled points out of a total %d points : %d" % (x_t.shape[0], numErr))
        end = time.time()

        x = np.linspace(1, self.numberOfSamples, len(acc))

        # print(x.shape, len(acc))
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.plot(x, acc, '-b')
        plt.subplot(2, 1, 2)
        plt.plot(x, kappa, '-g')
        print("Mean Accuracy is: %.2f +- %.2f" % (np.mean(acc) * 100, np.sqrt(np.var(acc))))
        print("Mean Kappa is: %.2f +- %.2f" % (np.mean(kappa) * 100, np.sqrt(np.var(kappa))))
        print('Executation Time: %.2f' % (end - start))
        print('Model Size: %.2f KB' % ((memory/N) / 1000))
        plt.ylim(ymin=0)
        plt.show()

        return

    def plotStream(self, stream: FileStream):

        X, y = stream.get_all_samples()

        z = [i for i, x in enumerate(y) if x == 0]
        X[z, :] = 0

        colNum = X.shape[1]
        X = np.reshape(X, (self.numberOfSamples * colNum))
        o = np.convolve(X, (np.ones(self.windowSize * colNum) / (self.windowSize * colNum)), mode='valid')

        plt.plot(np.linspace(1, len(o), num=len(o)), o)
        plt.show()
    def getEvaluationMetrics(self, evalPath : str):
        out = {}
        df = pd.read_csv(evalPath, comment='#')
        out[self.metricNames['ACC_MEAN']] = np.mean(df['current_acc_[M0]'])
        out[self.metricNames['ACC_STD']] = np.std(df['current_acc_[M0]'])
        out[self.metricNames['KAPPA_MEAN']] = np.mean(df['current_kappa_[M0]'])
        out[self.metricNames['KAPPA_STD']] = np.std(df['current_kappa_[M0]'])
        out[self.metricNames['EXEC_TIME']] = df['total_running_time_[M0]'].values[-1]
        out[self.metricNames['MODEL_SIZE']] = df['model_size_[M0]'].values[-1]
        return out