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
package ro.hasna.commons.weka.type;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * @since 0.1
 */
public class ValidationResult {
    protected final long trainingTime;
    protected final long testingTime;
    protected final double[][] confusionMatrix;
    protected Map<String, Object> metadata;

    /**
     * Create a tuple with validation result details.
     *
     * @param trainingTime    the time (in nanoseconds) taken for building the model using all the provided training instances
     * @param testingTime     the time (in nanoseconds) taken for predicting the class for only one test instance
     * @param confusionMatrix the confusion matrix of the model evaluation
     */
    public ValidationResult(long trainingTime, long testingTime, double[][] confusionMatrix) {
        this.trainingTime = trainingTime;
        this.testingTime = testingTime;
        this.confusionMatrix = confusionMatrix;
        this.metadata = null;
    }

    public long getTrainingTime() {
        return trainingTime;
    }

    public double[][] getConfusionMatrix() {
        return confusionMatrix;
    }

    public long getTestingTime() {
        return testingTime;
    }

    public void putMetadata(String key, Object value) {
        if (metadata == null) {
            metadata = new HashMap<>();
        }
        metadata.put(key, value);
    }

    public Object getMetadataValue(String key) {
        if (metadata == null) {
            return null;
        }
        return metadata.get(key);
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ValidationResult that = (ValidationResult) o;
        return trainingTime == that.trainingTime &&
                testingTime == that.testingTime &&
                Arrays.equals(confusionMatrix, that.confusionMatrix) &&
                Objects.equals(metadata, that.metadata);
    }

    @Override
    public int hashCode() {
        return Objects.hash(trainingTime, confusionMatrix, testingTime, metadata);
    }
}
