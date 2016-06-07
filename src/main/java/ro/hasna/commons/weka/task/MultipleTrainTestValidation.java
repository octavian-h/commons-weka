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
 * <p>
 * <pre>{@code
 *      Classifier classifier = ...
 *      Instances train = WekaUtils.readInstances("path/to/train.arff");
 *      Instances test = WekaUtils.readInstances("path/to/test.arff");
 *
 *      MultipleTrainTestValidation task = new MultipleTrainTestValidation.Builder(classifier, train, test)
 *                                        .trainSizePercentages(Arrays.asList(0.6, 0.7, 0.8))
 *                                        .iterations(10)
 *                                        .build();
 *      List<ValidationResult> results = task.call(); //the method blocks the current thread until it finishes the computation
 *
 *      ValidationResultWriter writer = new CsvValidationResultWriter("path/to/result.csv").build();
 *      writer.write(results);
 *      writer.close();
 * }</pre>
 *
 * @since 0.3
 */
public class MultipleTrainTestValidation implements Callable<List<ValidationResult>> {
    public static final String TRAIN_SIZE_PERCENTAGE_KEY = "train_size_percentage";
    public static final List<String> RESULT_METADATA_KEYS = Collections.singletonList(TRAIN_SIZE_PERCENTAGE_KEY);
    private static final Logger logger = Logger.getLogger(MultipleTrainTestValidation.class.getName());
    private final Classifier classifier;
    private final Instances trainInstances;
    private final Instances testInstances;
    private final ExecutorService executorService;
    private final boolean closeExecutorService;
    private final List<Double> trainSizePercentages;
    private final int iterations;

    private MultipleTrainTestValidation(Builder builder) {
        classifier = builder.classifier;
        trainInstances = builder.trainInstances;
        trainSizePercentages = builder.trainSizePercentages;
        testInstances = builder.testInstances;
        executorService = builder.executorService;
        closeExecutorService = builder.closeExecutorService;
        iterations = builder.iterations;
    }

    @Override
    public List<ValidationResult> call() throws Exception {
        int n = iterations * trainSizePercentages.size();
        List<ValidationResult> results = new ArrayList<>(n);
        List<Future<ValidationResult>> futures = new ArrayList<>(n);

        boolean done = false;
        try {
            final String trainName = trainInstances.relationName();
            final String testName = testInstances.relationName();

            for (int i = 0; i < iterations; i++) {
                final int iterationNumber = i + 1;

                // randomize train instances
                Instances trainInstancesCopy = new Instances(trainInstances);
                trainInstancesCopy.randomize(new Random(i));

                for (final double trainSizePercentage : trainSizePercentages) {
                    futures.add(executorService.submit(() -> {
                        logger.info(String.format("Evaluating train=%s (%.2f), test=%s, iteration=%d",
                                trainName, trainSizePercentage, testName, iterationNumber));

                        //copy the classifier so as to be used in parallel
                        Classifier classifierCopy = AbstractClassifier.makeCopy(classifier);

                        Instances train = WekaUtils.getTrainAndTestInstancesStratified(trainInstancesCopy, trainSizePercentage)[0];

                        ValidationResult validationResult = new TrainTestValidation(classifierCopy, train, testInstances).call();
                        validationResult.putMetadata(TRAIN_SIZE_PERCENTAGE_KEY, trainSizePercentage);
                        return validationResult;
                    }));
                }
            }

            // await till we get all results
            for (Future<ValidationResult> future : futures) {
                results.add(future.get());
            }

            // set the flag that we finished properly
            done = true;

            return results;
        } finally {
            if (!done) {
                for (Future<ValidationResult> future : futures) {
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
        private ExecutorService executorService;
        private boolean closeExecutorService;
        private List<Double> trainSizePercentages;
        private int iterations;

        /**
         * Constructor for building the class MultipleTrainTestValidation.
         *
         * @param classifier     the Weka classifier
         * @param trainInstances the set of train instances
         * @param testInstances  the set of test instances
         */
        public Builder(Classifier classifier, Instances trainInstances, Instances testInstances) {
            this.classifier = classifier;
            this.trainInstances = trainInstances;
            this.testInstances = testInstances;

            // default values
            this.trainSizePercentages = Collections.singletonList(1.0);
            this.iterations = 1;
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

        public MultipleTrainTestValidation build() {
            if (this.executorService == null) {
                this.executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
            }

            return new MultipleTrainTestValidation(this);
        }
    }
}
