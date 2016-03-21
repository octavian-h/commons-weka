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

import weka.classifiers.Evaluation;

import java.util.Objects;

/**
 * @since 0.1
 */
public class ValidationResult {
    protected final long modelBuildingTime;
    protected final Evaluation predictingResult;
    protected final long predictingTime;

    /**
     * Create a tuple with validation result details.
     *
     * @param modelBuildingTime the time (in nanoseconds) taken for building the mode using all the provided training instances
     * @param predictingResult  the result of the model predictions
     * @param predictingTime    the time (in nanoseconds) taken for predicting the class for only one test instance
     */
    public ValidationResult(long modelBuildingTime, Evaluation predictingResult, long predictingTime) {
        this.modelBuildingTime = modelBuildingTime;
        this.predictingResult = predictingResult;
        this.predictingTime = predictingTime;
    }

    public long getModelBuildingTime() {
        return modelBuildingTime;
    }

    public Evaluation getPredictingResult() {
        return predictingResult;
    }

    public long getPredictingTime() {
        return predictingTime;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ValidationResult that = (ValidationResult) o;
        return modelBuildingTime == that.modelBuildingTime &&
                predictingTime == that.predictingTime &&
                Objects.equals(predictingResult, that.predictingResult);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelBuildingTime, predictingResult, predictingTime);
    }
}
