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

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import ro.hasna.commons.weka.type.ValidationResult;
import ro.hasna.commons.weka.util.WekaUtils;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

import java.nio.file.Paths;

/**
 * @since 0.4
 */
public class ClusterValidationTest {
    private Clusterer clusterer;
    private Instances data;
    private ClusterValidation task;

    @Before
    public void setUp() throws Exception {
        clusterer = new SimpleKMeans();
        data = WekaUtils.readInstances(Paths.get(getClass().getResource("/iris.arff").toURI()));
        task = new ClusterValidation(clusterer, data);
    }

    @After
    public void tearDown() throws Exception {
        task = null;
        data = null;
        clusterer = null;
    }

    @Test
    public void testCompleteRun() throws Exception {
        ValidationResult validationResult = task.call();

        double[][] confusionMatrix = validationResult.getConfusionMatrix();

        Assert.assertEquals(data.numClasses(), confusionMatrix.length);
        Assert.assertTrue(confusionMatrix[0].length >= data.numClasses());
    }

    @Test
    public void testRemoveInstancesClasses() throws Exception {
        ClusterValidation task = new ClusterValidation(clusterer, data);
        Instances unlabelledInstances = task.removeInstancesClasses();

        Assert.assertEquals(data.numAttributes() - 1, unlabelledInstances.numAttributes());
    }

    @Test
    public void testComputeClusterClass() throws Exception {
        double[][] matrix = {
                {20, 21, 24, 20, 14},
                {18, 20, 20, 20, 20},
                {22, 19, 16, 20, 26}
        };

        int[] expected = {-1, 1, 0, -1, 2};
        int[] predicted = task.computeClusterClass(3, 5, matrix);

        Assert.assertArrayEquals(expected, predicted);
    }

    @Test
    public void testReorderClustersMoreClusters() throws Exception {
        double[][] matrix = {
                {20, 21, 24, 20, 14},
                {18, 20, 20, 20, 20},
                {22, 19, 16, 20, 26}
        };
        int[] clusterClass = {-1, 1, 0, -1, 2};

        double[][] expectedMatrix = {
                {24, 21, 14, 20, 20},
                {20, 20, 20, 20, 18},
                {16, 19, 26, 20, 22}
        };
        task.reorderClustersInConfusionMatrix(3, 5, matrix, clusterClass);

        for (int i = 0; i < expectedMatrix.length; i++) {
            double[] expectedRow = expectedMatrix[i];
            double[] predictedRow = matrix[i];
            Assert.assertArrayEquals(expectedRow, predictedRow, 0.1);
        }
    }

    @Test
    public void testReorderClustersMoreClasses() throws Exception {
        double[][] matrix = {
                {20, 21, 0},
                {18, 20, 0},
                {22, 19, 0}
        };
        int[] clusterClass = {2, 0};

        double[][] expectedMatrix = {
                {21, 0, 20},
                {20, 0, 18},
                {19, 0, 22}
        };
        task.reorderClustersInConfusionMatrix(3, 2, matrix, clusterClass);

        for (int i = 0; i < expectedMatrix.length; i++) {
            double[] expectedRow = expectedMatrix[i];
            double[] predictedRow = matrix[i];
            Assert.assertArrayEquals(expectedRow, predictedRow, 0.1);
        }
    }
}