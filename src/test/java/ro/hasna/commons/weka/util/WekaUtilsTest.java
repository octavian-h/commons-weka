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
}