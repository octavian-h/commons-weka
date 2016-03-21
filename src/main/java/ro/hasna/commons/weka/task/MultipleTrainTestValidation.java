/*
 * Copyright (C) 2016 Octavian Hasna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ro.hasna.commons.weka.task;

import ro.hasna.commons.weka.io.ValidationResultWriter;
import ro.hasna.commons.weka.type.ValidationResult;
import ro.hasna.commons.weka.util.WekaUtils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Logger;

/**
 * Task for running the validation on multiple train set sizes.
 *
 * <pre>{@code
 *      Classifier classifier = ...
 *      Instances train = WekaUtils.readInstances("path/to/train.arff");
 *      Instances test = WekaUtils.readInstances("path/to/test.arff");
 *
 *      try(ValidationResultWriter writer = new CsvWriter("path/to/result.csv").build()){
 *          MultipleTrainTestValidation task = new MultipleTrainTestValidation.Builder(classifier, train, test, writer)
 *                                              .trainSizePercentages(Arrays.asList(0.6, 0.7, 0.8))
 *                                              .folds(5)
 *                                              .iterations(10)
 *                                              .build();
 *          task.call(); //the method block the current thread until it finish
 *      }
 * }</pre>
 *
 * @since 0.1
 */
public class MultipleTrainTestValidation implements Callable<Boolean> {
    private final static Logger logger = Logger.getLogger(MultipleTrainTestValidation.class.getName());
    private final Classifier classifier;
    private final Instances trainInstances;
    private final Instances testInstances;
    private final ValidationResultWriter validationResultWriter;
    private final List<String> extraColumns;
    private final ExecutorService executorService;
    private final boolean closeExecutorService;
    private final List<Double> trainSizePercentages;
    private final int folds;
    private final int iterations;

    private MultipleTrainTestValidation(Builder builder) {
        classifier = builder.classifier;
        trainInstances = builder.trainInstances;
        trainSizePercentages = builder.trainSizePercentages;
        testInstances = builder.testInstances;
        folds = builder.folds;
        validationResultWriter = builder.validationResultWriter;
        extraColumns = builder.extraColumns;
        executorService = builder.executorService;
        closeExecutorService = builder.closeExecutorService;
        iterations = builder.iterations;
    }

    private static Instances getRandomizedAndStratifiedInstances(Instances instances, Random random, int folds) {
        final Instances trainInstancesCopy = new Instances(instances);
        trainInstancesCopy.randomize(random);
        if (folds > 1) {
            trainInstancesCopy.stratify(folds);
        }
        return trainInstancesCopy;
    }

    @Override
    public Boolean call() throws Exception {
        List<Future<Boolean>> futures = new ArrayList<>();
        boolean done = false;
        try {
            final String trainName = trainInstances.relationName();
            final String testName = testInstances.relationName();

            for (int i = 0; i < iterations; i++) {
                final int iterationNumber = i + 1;

                // randomize train instances
                Instances trainInstancesCopy = getRandomizedAndStratifiedInstances(trainInstances, new Random(i), folds);

                // randomize test instances
                Instances testInstancesCopy = getRandomizedAndStratifiedInstances(testInstances, new Random(i), folds);

                for (int j = 0; j < folds; j++) {
                    final int foldNumber = j + 1;

                    final Instances trainInstancesFold = folds > 1 ? trainInstancesCopy.trainCV(folds, j) : trainInstancesCopy;
                    final Instances testInstancesFold = folds > 1 ? testInstancesCopy.testCV(folds, j) : testInstancesCopy;

                    for (final double trainSizePercentage : trainSizePercentages) {
                        futures.add(executorService.submit(() -> {
                            logger.config(String.format("Evaluating train=%s (%.2f), test=%s, fold=%d, iteration=%d",
                                    trainName, trainSizePercentage, testName, foldNumber, iterationNumber));

                            //copy the classifier so as to be used in parallel
                            Classifier classifierCopy = AbstractClassifier.makeCopy(classifier);

                            Instances train = WekaUtils.getTrainAndTestInstancesStratified(trainInstancesFold, trainSizePercentage)[0];
                            ValidationResult result = new TrainTestValidation(classifierCopy, train, testInstancesFold).call();

                            validationResultWriter.write(classifierCopy, train, testInstancesFold,
                                    trainSizePercentage, foldNumber, iterationNumber, extraColumns,
                                    result);

                            return true;
                        }));
                    }
                }
            }

            // await till we get all results
            for (Future future : futures) {
                future.get();
            }

            // set the flag that we finished properly
            done = true;

            return true;
        } finally {
            if (!done) {
                for (Future<Boolean> future : futures) {
                    future.cancel(true);
                }
            }

            if (closeExecutorService) {
                executorService.shutdown();
            }
        }
    }

    public static class Builder {
        private final Classifier classifier;
        private final Instances trainInstances;
        private final Instances testInstances;
        private final ValidationResultWriter validationResultWriter;
        private List<String> extraColumns;
        private ExecutorService executorService;
        private boolean closeExecutorService;
        private List<Double> trainSizePercentages;
        private int folds;
        private int iterations;

        /**
         * Constructor for building the class MultipleTrainTestValidation.
         *
         * @param classifier             the Weka classifier
         * @param trainInstances         the set of train instances
         * @param testInstances          the set of test instances
         * @param validationResultWriter the writer for the validation results
         */
        public Builder(Classifier classifier, Instances trainInstances, Instances testInstances,
                       ValidationResultWriter validationResultWriter) {
            this.classifier = classifier;
            this.trainInstances = trainInstances;
            this.testInstances = testInstances;
            this.validationResultWriter = validationResultWriter;

            // default values
            this.trainSizePercentages = Collections.singletonList(1.0);
            this.iterations = 1;
            this.folds = 1;
            this.extraColumns = Collections.emptyList();
            this.closeExecutorService = true;
        }


        public Builder trainSizePercentages(List<Double> percentages) {
            if (percentages.isEmpty()) {
                throw new IllegalArgumentException("percentages list is empty");
            }
            for (double percentage : percentages) {
                if (percentage <= 0 || percentage > 1) {
                    throw new IllegalArgumentException("percentages must be between 0 (exclusive) and 1 (inclusive)");
                }
            }

            this.trainSizePercentages = percentages;
            return this;
        }

        public Builder folds(int folds) {
            if (folds <= 0) {
                throw new IllegalArgumentException("folds must be positive");
            }

            this.folds = folds;
            return this;
        }

        public Builder iterations(int iterations) {
            if (iterations <= 0) {
                throw new IllegalArgumentException("iterations must be positive");
            }

            this.iterations = iterations;
            return this;
        }

        public Builder executorService(ExecutorService executorService) {
            this.executorService = executorService;
            return this;
        }

        public Builder closeExecutorService(boolean closeExecutorService) {
            this.closeExecutorService = closeExecutorService;
            return this;
        }

        public Builder extraColumns(List<String> extraColumns) {
            this.extraColumns = extraColumns;
            return this;
        }

        public MultipleTrainTestValidation build() {
            if (this.executorService == null) {
                this.executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
            }

            return new MultipleTrainTestValidation(this);
        }
    }
}
