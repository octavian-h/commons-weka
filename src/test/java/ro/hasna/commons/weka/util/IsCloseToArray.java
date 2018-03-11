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

import org.hamcrest.Description;
import org.hamcrest.Factory;
import org.hamcrest.Matcher;
import org.hamcrest.TypeSafeMatcher;

/**
 * @since 0.5
 */
public class IsCloseToArray extends TypeSafeMatcher<double[]> {
    private final double[] expectedArray;
    private final double delta;

    public IsCloseToArray(double[] expectedArray, double delta) {
        this.expectedArray = expectedArray;
        this.delta = delta;
    }

    @Factory
    public static Matcher<double[]> closeToArray(double[] operand, double error) {
        return new IsCloseToArray(operand, error);
    }

    @Override
    public void describeTo(Description description) {
        description.appendText("a numeric array within ").appendValue(delta).appendText(" of ").appendValue(expectedArray);
    }

    @Override
    protected boolean matchesSafely(double[] actualArray) {
        if (expectedArray.length != actualArray.length) {
            return false;
        }
        for (int i = 0; i < expectedArray.length; i++) {
            if (Math.abs(expectedArray[i] - actualArray[i]) - delta > 0) {
                return false;
            }
        }

        return true;
    }
}
