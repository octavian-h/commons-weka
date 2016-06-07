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

import java.io.Flushable;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * @since 0.3
 */
public interface ValidationResultWriter extends AutoCloseable, Flushable {

    /**
     * Write a list of validation results with the given shared metadata.
     *
     * @param results        the list of validation results
     * @param sharedMetadata metadata that is common for all the results
     * @throws IOException if the validation results could not be written
     */
    void write(List<ValidationResult> results, Map<String, Object> sharedMetadata) throws IOException;

    /**
     * Write a list of validation results.
     *
     * @param results the list of validation results
     * @throws IOException if the validation results could not be written
     */
    default void write(List<ValidationResult> results) throws IOException {
        write(results, Collections.emptyMap());
    }

    /**
     * Write the validation result with the given shared metadata.
     *
     * @param result         the validation result
     * @param sharedMetadata metadata that is common for all the results
     * @throws IOException if the validation result could not be written
     */
    default void write(ValidationResult result, Map<String, Object> sharedMetadata) throws IOException {
        write(Collections.singletonList(result), sharedMetadata);
    }

    /**
     * Write the validation result.
     *
     * @param result the validation result
     * @throws IOException if the validation result could not be written
     */
    default void write(ValidationResult result) throws IOException {
        write(Collections.singletonList(result), Collections.emptyMap());
    }
}
