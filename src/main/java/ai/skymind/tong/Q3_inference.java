package ai.skymind.tong;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.ImagePreProcessingSerializerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;

import java.io.File;
import java.util.Random;

public class Q3_inference {

    public static void main(String[] args) throws Exception {
        //load test dataset
        String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        Random randNumGen = new Random(1234);

        File parentDir = new File("C:\\Users\\user\\Desktop\\Penjana_DLPC\\Day11\\Q3\\natural_images\\seg_test");
        FileSplit fileSplit = new FileSplit(parentDir, allowedExtensions, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        //no data transformation , no splitting
        int height = 150;
        int width = 150;
        int channels = 3;

        //record reader
        ImageRecordReader testRR = new ImageRecordReader(height,width,channels,labelMaker);
        testRR.initialize(fileSplit);

        //3.1.5 iterator
        int labelIndex  = 1; //for images, trainRR.next() = List{NDArray image, int label}
        int numLabels = testRR.numLabels();
        int batchSize  = 6;
        DataSetIterator test_iter = new RecordReaderDataSetIterator(testRR, batchSize, labelIndex, numLabels);

        //load normalizer
        NormalizerSerializer normalizerSerializer = new NormalizerSerializer().addStrategy(new ImagePreProcessingSerializerStrategy());
        ImagePreProcessingScaler scaler = normalizerSerializer.restore("C:/Users/user/Desktop/Penjana_DLPC/Day11/Q3/normalizer.zip");
        test_iter.setPreProcessor(scaler);

        //load model
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("C:/Users/user/Desktop/Penjana_DLPC/Day11/Q3/model.zip", true);

        //testing
        model.fit(test_iter);
        Evaluation evalTest = model.evaluate(test_iter);  //validation with training data
        System.out.println("Test: " + evalTest.stats());

    }
}
