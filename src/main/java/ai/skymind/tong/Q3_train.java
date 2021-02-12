package ai.skymind.tong;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

public class Q3_train {
    //Images are of format given by allowedExtension
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private static final long seed = 12345;
    private static final Random randNumGen = new Random(seed);

    private static final int height = 150;
    private static final int width = 150;
    private static final int channels = 3;

    public static void main(String[] args) throws Exception {

        // define image folder location
        File parentDir = new File("C:/Users/user/Desktop/Penjana_DLPC/Day11/Q3/natural_images/seg_train/seg_train");

        //use file with specified extensions
        FileSplit fileSplit = new FileSplit(parentDir, allowedExtensions, randNumGen);

        ///label from directories
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        //sampling for training & testing
        InputSplit[] filesInDirSplit = fileSplit.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        //resize all images to specified dimensions
        ImageRecordReader trainRR = new ImageRecordReader(height,width,channels,labelMaker);
        ImageRecordReader testRR = new ImageRecordReader(height,width,channels,labelMaker);

        //Define image transformation
        FlipImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(5);
        ImageTransform rotateImage = new RotateImageTransform(randNumGen, 15);
        //ImageTransform showImage = new ShowImageTransform("Image",1000);
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip,0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage,0.3)//,
                //new Pair<>(showImage,1.0)
        );
        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);

        //Initialize the record reader with the train data and the transform chain
        trainRR.initialize(trainData,transform);
        testRR.initialize(testData);  //not transforming testing

        //training
        Training(trainRR, testRR);
    }

    public static void Training(ImageRecordReader trainRR, ImageRecordReader testRR)
    {
        int numLabels = trainRR.numLabels();
        System.out.println("No. of classes = " + numLabels);
        int labelIndex = 1; //trainRR.next() = List{NDArray image, int label}
        int batchSize = 10;

        //make iterators from RR
        DataSetIterator train_iter = new RecordReaderDataSetIterator(trainRR, batchSize, labelIndex, numLabels);
        DataSetIterator test_iter = new RecordReaderDataSetIterator(testRR, batchSize, labelIndex, numLabels);

        //normalize input???
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        train_iter.setPreProcessor(scaler);
        test_iter.setPreProcessor(scaler);  //apply on both

        HashMap<Integer, Double> schedule = new HashMap<>();
        schedule.put(0, 0.001);
        schedule.put(8, 1e-4);
        schedule.put(12, 1e-5);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)  //weight initialization
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH, schedule)))  //optimizer
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.RELU)  //default activation function
                .miniBatch(true)
                //.l2(0.00001)
                .list()
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .stride(2, 2)
                        .nOut(256)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .poolingType(PoolingType.MAX)
                        .stride(2, 2)
                        .build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .stride(2, 2)
                        .nOut(256)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .poolingType(PoolingType.MAX)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(1, 1)
                        //.convolutionMode(ConvolutionMode.Same)
                        .stride(2, 2)
                        .nOut(127)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(55)
                        .build())
                .layer(new OutputLayer.Builder()  //output layer
                        .lossFunction(LossFunctions.LossFunction.MCXENT)  //multiclass cross-entropy loss
                        .activation(Activation.SOFTMAX)
                        .nOut(train_iter.totalOutcomes())  //specifiy the class nunmber for datasets
                        .build())
                .setInputType(InputType.convolutional(height, width, train_iter.inputColumns())) // InputType.convolutional for normal image
                .backpropType(BackpropType.Standard)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config); //construct model
        model.init();  //initialize model

        //set up server to record training statistics
        InMemoryStatsStorage storage = new InMemoryStatsStorage(); //allocate some memory
        UIServer server = UIServer.getInstance();
        server.attach(storage);  //attach memory to use for server
        model.setListeners(new StatsListener(storage), new ScoreIterationListener(500));//*train_iter.batch()));  //print per 1000 iteration

        //training model
        int epoch = 20;
        for (int i = 0; i < epoch; ++i)
        {
            model.fit(train_iter);
            Evaluation evalTrain = model.evaluate(train_iter);  //validation with training data
            System.out.println("\nepoch = " + i);
            System.out.println("Training accuracy: " + evalTrain.accuracy());
            Evaluation evalTest = model.evaluate(test_iter);  //test with testing data
            System.out.println("Validation  accuracy: " + evalTest.accuracy());
        }

        Evaluation evalTrain = model.evaluate(train_iter);  //validation with training data
        System.out.println("\nTraining: " + evalTrain.stats());
        Evaluation evalTest = model.evaluate(test_iter);  //test with testing data
        System.out.println("Test: " + evalTest.stats());

    }
}
