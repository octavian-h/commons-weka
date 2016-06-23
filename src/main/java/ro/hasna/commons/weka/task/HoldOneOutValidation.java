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
 *      Instances instances = WekaUtils.readInstances("path/to/data.arff");
 *      ValidationResult result = new HoldOneOutValidation(classifier, instances).call();
 *
 *      ValidationResultWriter writer = new CsvValidationResultWriter("path/to/result.csv").build();
 *      writer.write(result);
 *      writer.close();
 * }</pre>
 *
 * @since 0.1
 */
public class HoldOneOutValidation implements Callable<ValidationResult> {
    private final Classifier classifier;
    private final Instances instances;

    public HoldOneOutValidation(Classifier classifier, Instances instances) {
        this.classifier = classifier;
        this.instances = instances;
    }

    @Override
    public ValidationResult call() throws Exception {
        Evaluation evaluation = new Evaluation(instances);

        long modelBuildingTime = 0L;
        long predictingTime = 0L;

        Instances copy = new Instances(instances);
        for (int i = 0; i < instances.size(); i++) {
            if (i == 0) {
                copy.remove(i);
            } else {
                copy.set(i - 1, instances.get(i - 1)); //remove+add
            }

            long startTime = System.nanoTime();
            classifier.buildClassifier(copy);
            modelBuildingTime += System.nanoTime() - startTime;

            startTime = System.nanoTime();
            evaluation.evaluateModelOnceAndRecordPrediction(classifier, instances.get(i));
            predictingTime += (System.nanoTime() - startTime - predictingTime) / (i + 1); //iterative moving average
        }

        return new ValidationResult(modelBuildingTime, predictingTime, evaluation.confusionMatrix());
    }
}
