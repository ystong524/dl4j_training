package ai.skymind.nd4j_ex;
/*
 * Copyright (c) 2019 Skymind Holdings Bhd.
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class nd4j_ex6 {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        int nRows = 3;
        int nColumns = 5;
        INDArray shape = Nd4j.create(new int[]{nRows, nColumns});
        INDArray myArray = Nd4j.rand(shape,123);
        System.out.println(BLACK_BOLD + "Default array" + ANSI_RESET);
        System.out.println(myArray);

        //array log
        INDArray logArray = Transforms.log(myArray);
        System.out.println(BLACK_BOLD + "\nArray log transform" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Transforms.log(myArray)" + ANSI_RESET);
        System.out.println(logArray);

        //array absolute value
        INDArray absArray = Transforms.abs(logArray);
        System.out.println(BLACK_BOLD + "\nArray absolute transform" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Transforms.abs(myArray)" + ANSI_RESET);
        System.out.println(absArray);

        //Round up array
        INDArray roundUpArray = Transforms.ceil(absArray);
        System.out.println(BLACK_BOLD + "\nRound up array" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Transforms.ceil(absArray)" + ANSI_RESET);
        System.out.println(roundUpArray);

        //Array sigmoid function
        INDArray sigmoidArray = Transforms.sigmoid(myArray);
        System.out.println(BLACK_BOLD + "\nArray sigmoid function" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Transforms.sigmoid(myArray)" + ANSI_RESET);
        System.out.println(sigmoidArray);

        // For more operation: https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ops/TransformOp.html

        /*
        EXERCISE:
        - Create arr1 with shape(3,3) initialize with random value
        - Perform TanH operation on arr1
        - Perform round operation on arr1
        */
        INDArray res = Transforms.isMax(myArray);
        System.out.println(res);

        INDArray arr1, arr2, arr3;
        arr1 = Nd4j.rand(3,3);
        System.out.println(arr1);
        arr2 = Transforms.sigmoid(arr1, false);
        System.out.println(arr2);
        arr3 = Transforms.round(arr1, false);
        System.out.println(arr3);

    }
}
