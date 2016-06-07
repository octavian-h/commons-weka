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
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.concurrent.Callable;

/**
 * Task for running a simple validation.
 * <p>
 * <pre>{@code
 *      Classifier classifier = ...
 *      Instances train = WekaUtils.readInstances("path/to/train.arff");
 *      Instances test = WekaUtils.readInstances("path/to/test.arff");
 *      ValidationResult result = new TrainTestValidation(classifier, train, test).call();
 *
 *      ValidationResultWriter writer = new CsvValidationResultWriter("path/to/result.csv").build();
 *      writer.write(result);
 *      writer.close();
 * }</pre>
 *
 * @since 0.1
 */
public class TrainTestValidation implements Callable<ValidationResult> {
    private final Classifier classifier;
    private final Instances trainInstances;
    private final Instances testInstances;

    public TrainTestValidation(Classifier classifier, Instances trainInstances, Instances testInstances) {
        this.classifier = classifier;
        this.trainInstances = trainInstances;
        this.testInstances = testInstances;
    }

    @Override
    public ValidationResult call() throws Exception {
        Evaluation evaluation = new Evaluation(trainInstances);

        long startTime = System.nanoTime();
        classifier.buildClassifier(trainInstances);
        long modelBuildingTime = System.nanoTime() - startTime;

        startTime = System.nanoTime();
        evaluation.evaluateModel(classifier, testInstances);
        long predictingTime = (System.nanoTime() - startTime) / testInstances.size();

        return new ValidationResult(modelBuildingTime, predictingTime, evaluation.confusionMatrix());
    }
}
