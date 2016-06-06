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

import org.hamcrest.core.StringContains;
import org.junit.*;
import org.junit.rules.ExpectedException;
import ro.hasna.commons.weka.type.ValidationResult;
import ro.hasna.commons.weka.util.WekaUtils;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.nio.file.Paths;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * @since 0.3
 */
public class MultipleTrainTestValidationTest {
    @Rule
    public ExpectedException thrown = ExpectedException.none();
    private Classifier classifier;
    private Instances train;
    private Instances test;

    @Before
    public void setUp() throws Exception {
        classifier = new IBk(1);

        Instances instances = WekaUtils.readInstances(Paths.get(getClass().getResource("/iris.arff").toURI()));
        Instances[] instancesStratified = WekaUtils.getTrainAndTestInstancesStratified(instances, 0.8);

        train = instancesStratified[0];
        test = instancesStratified[1];
    }

    @After
    public void tearDown() throws Exception {
        classifier = null;
        train = null;
        test = null;
    }

    @Test
    public void testCompleteRun() throws Exception {
        NumberFormat numberFormat = NumberFormat.getNumberInstance(Locale.ENGLISH);

        MultipleTrainTestValidation task = new MultipleTrainTestValidation.Builder(classifier, train, test)
                .trainSizePercentages(Arrays.asList(0.6, 0.7, 0.8))
                .iterations(5)
                .build();
        List<ValidationResult> validationResults = task.call();

        Assert.assertEquals(15, validationResults.size());//iterations * percentages
        for (ValidationResult validationResult : validationResults) {
            Assert.assertEquals(1, validationResult.getMetadata().size());
            String str = validationResult.getMetadataValue(MultipleTrainTestValidation.TRAIN_SIZE_PERCENTAGE_KEY).toString();
            int percentage = (int) (numberFormat.parse(str).doubleValue() * 10);
            Assert.assertTrue(6 <= percentage && percentage <= 8); //train_size_percentage
        }
    }

    @Test
    public void testWrongNumberOfIterations() throws Exception {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("iterations must be positive");

        new MultipleTrainTestValidation.Builder(classifier, train, test)
                .iterations(0)
                .build();
    }

    @Test
    public void testEmptyPercentagesList() throws Exception {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("percentages list is empty");

        new MultipleTrainTestValidation.Builder(classifier, train, test)
                .trainSizePercentages(Collections.emptyList())
                .build();
    }

    @Test
    public void testWrongPercentagesValues() throws Exception {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("percentages must be between 0 (exclusive) and 1 (inclusive)");

        new MultipleTrainTestValidation.Builder(classifier, train, test)
                .trainSizePercentages(Collections.singletonList(0.0))
                .build();
    }

    @Test
    public void testWrongClassifier() throws Exception {
        thrown.expect(ExecutionException.class);
        thrown.expectMessage(StringContains.containsString("Cannot handle multi-valued nominal class!"));

        ExecutorService executorService = Executors.newFixedThreadPool(2);
        LinearRegression regression = new LinearRegression();
        MultipleTrainTestValidation task = new MultipleTrainTestValidation.Builder(regression, train, test)
                .executorService(executorService)
                .closeExecutorService(false)
                .iterations(10)
                .build();

        task.call();
        executorService.shutdown();
    }
}