package ai.skymind.DataVec;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.datavec.api.records.reader.RecordReader;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class load_csv {

    private static int linetoskip = 0;
    private static char delimiter = ',';
    private static int batch_num = 5;


    //1. filesplit
    //2.record reader
    //3. iterator
    //4. dataset --getFeatures().shape()

    public static void main(String[] args) throws IOException, InterruptedException {
        /*//csv file location
        //folder path is root
        String root = System.getProperty("user.dir");
        System.out.println(root);
        Path f1 = Paths.get("src/main/resources/datavec/iris.txt");

        Stream<String> lines = Files.lines(f1, Charset.defaultCharset());
        int num_lines = (int) lines.count();
        System.out.println(num_lines);

        // use ClassPathResource will direct to "resources"
        File f = new ClassPathResource("datavec/iris.txt").getFile();
        FileSplit fs = new FileSplit(f);

        //record reader for handling loading/parsing
        RecordReader rr = new CSVRecordReader(linetoskip, delimiter);
        rr.initialize(fs);*/


        //other known properties about dataset
        int label_idx = 4;
        int classes = 3;



        File inputFile = new ClassPathResource("datavec/iris.txt").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);

        // get dataset using record reader. CSVRecordReader handles loading/parsing
        RecordReader recordReader = new CSVRecordReader(linetoskip, delimiter);
        recordReader.initialize(fileSplit);


        //create iterator && loop all to a dataset
        DataSetIterator iterator1 = new RecordReaderDataSetIterator(recordReader, 150, label_idx, classes);
        DataSet allData = iterator1.next();

        System.out.println("shape:\n" + allData.getFeatures().shape().toString());


        /*////-----------------------manipulate----------------------////
        //shuffle
        allData.shuffle();

        // train-test split
        SplitTestAndTrain test_train_split = allData.splitTestAndTrain(0.6);

        //make new dataset
        DataSet train = test_train_split.getTrain();
        DataSet test = test_train_split.getTest();

        //make new iterator
        DataSetIterator train_iter = new ViewIterator(train, batch_num);
        DataSetIterator test_iter = new ViewIterator(test, batch_num);

        //normalization
        DataNormalization scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(train_iter);  //fit it for training ******iterator
        //apply it to preprocess both
        train_iter.setPreProcessor(scaler);
        test_iter.setPreProcessor(scaler);

        while (train_iter.hasNext())
        {
            System.out.println("scaled training\n" + train_iter.next());
        }
        while (test_iter.hasNext())
        {
            System.out.println("scaled testing\n" + test_iter.next());
        }*/






    }
}
