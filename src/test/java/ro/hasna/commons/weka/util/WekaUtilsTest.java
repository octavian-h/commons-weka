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
package ro.hasna.commons.weka.util;

import org.junit.Assert;
import org.junit.Test;
import weka.core.Instance;
import weka.core.Instances;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

/**
 * @since 0.1
 */
public class WekaUtilsTest {
    @Test
    public void testReadAndWriteInstances() throws Exception {
        Instances instances = WekaUtils.readInstances(Paths.get(getClass().getResource("/iris.arff").toURI()));
        Path tmpOutputPath = Paths.get("iris_tmp.arff");
        WekaUtils.writeInstances(instances, tmpOutputPath);
        Instances copy = WekaUtils.readInstances(tmpOutputPath);

        Files.delete(tmpOutputPath);

        Assert.assertEquals(instances.relationName(), copy.relationName());
        Assert.assertTrue(copy.equalHeaders(instances));
        Assert.assertEquals(instances.size(), copy.size());
    }

    @Test
    public void testDivideInstances() throws Exception {
        Instances instances = WekaUtils.readInstances(Paths.get(getClass().getResource("/iris.arff").toURI()));
        Instances[] v = WekaUtils.getTrainAndTestInstancesStratified(instances, 0.8);

        Assert.assertEquals(v[0].size(), v[1].size() * 4);

        int trainSize = 0;
        for (Instance instance : v[0]) {
            if (instance.classValue() == 0) {
                trainSize++;
            }
        }

        int testSize = 0;
        for (Instance instance : v[1]) {
            if (instance.classValue() == 0) {
                testSize++;
            }
        }

        Assert.assertEquals(trainSize, testSize * 4);
    }

    @Test
    public void testBalanceInstances() throws Exception {
        Instances instances = WekaUtils.readInstances(Paths.get(getClass().getResource("/iris.arff").toURI()));
        instances.remove(0);
        instances.remove(1);
        instances.remove(2);
        Instances balancedInstances = WekaUtils.getBalancedInstances(instances);

        Map<Double, Integer> classesDistribution = WekaUtils.getClassesDistribution(balancedInstances);
        Assert.assertEquals(47, classesDistribution.get(0d).intValue());
        Assert.assertEquals(47, classesDistribution.get(1d).intValue());
        Assert.assertEquals(47, classesDistribution.get(2d).intValue());
    }

    @Test
    public void testGetIncorrectPercentage() throws Exception {
        double[][] matrix = {
                {2, 0, 0},
                {0, 4, 0},
                {5, 0, 14}
        };
        double incorrectPercentage = WekaUtils.getIncorrectPercentage(matrix);

        Assert.assertEquals(0.2, incorrectPercentage, 0.0001);
    }

    @Test
    public void testGetIncorrectPercentageMoreClusters() throws Exception {
        double[][] matrix = {
                {2, 0, 0, 2},
                {0, 4, 0, 1},
                {5, 0, 9, 2}
        };
        double incorrectPercentage = WekaUtils.getIncorrectPercentage(matrix);

        Assert.assertEquals(0.4, incorrectPercentage, 0.0001);
    }

    @Test
    public void testGetIncorrectPercentageMoreClasses() throws Exception {
        double[][] matrix = {
                {2, 0, 0},
                {0, 4, 0},
                {5, 0, 9},
                {2, 1, 2}
        };
        double incorrectPercentage = WekaUtils.getIncorrectPercentage(matrix);

        Assert.assertEquals(0.4, incorrectPercentage, 0.0001);
    }

    @Test
    public void testMultipleDivideInstances() throws Exception {
        Instances instances = WekaUtils.readInstances(Paths.get(getClass().getResource("/iris.arff").toURI()));
        Instances[] v = WekaUtils.getSetsOfInstancesStratified(instances, 10);

        Assert.assertEquals(v.length, 10);

        for (int i = 0; i < 10; i++) {
            int f[] = new int[3];
            for (Instance instance : v[i]) {
                f[(int) instance.classValue()]++;
            }

            Assert.assertEquals(f[0], 5);
            Assert.assertEquals(f[1], 5);
            Assert.assertEquals(f[2], 5);
        }
    }
}