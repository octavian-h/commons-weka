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
package ro.hasna.commons.weka.io;

import ro.hasna.commons.weka.type.ValidationResult;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.io.Flushable;
import java.io.IOException;
import java.util.List;

/**
 * @since 0.1
 */
public interface ValidationResultWriter extends AutoCloseable, Flushable {

    /**
     * Write the validation result.
     *
     * @param classifier          the classifier used for building the model
     * @param trainInstances      the train instances
     * @param testInstances       the test instances
     * @param trainSizePercentage the percentage of train instances used
     * @param foldNumber          the fold number
     * @param iterationNumber     the iteration number
     * @param extraColumns        extra columns (ex: test instances properties)
     * @param result              the validation result
     * @throws IOException if the validation result could not be written
     */
    void write(Classifier classifier, Instances trainInstances, Instances testInstances,
               double trainSizePercentage, int foldNumber, int iterationNumber, List<String> extraColumns,
               ValidationResult result) throws IOException;
}
