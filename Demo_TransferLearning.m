net = alexnet;
% You can also grab AlexNet from add-on explorer in the home tab, or file exchange online

layers = net.Layers;
layers 
% notice the 1000 in the last fully connected layer. This is for the 1000 categories AlexNet knows.

rootFolder = 'cifar10Train';
categories = {'Airplane','Automobile'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds = splitEachLabel(imds, 500, 'randomize') % we only need 500 images per class
imds.ReadFcn = @readFunctionTrain;

layers = layers(1:end-3);
layers(end+1) = fullyConnectedLayer(64, 'Name', 'special_2');
layers(end+1) = reluLayer;
layers(end+1) = fullyConnectedLayer(2, 'Name', 'fc8_2 ');
layers(end+1) = softmaxLayer;
layers(end+1) = classificationLayer();

layers(end-2).WeightLearnRateFactor = 10;
layers(end-2).WeightL2Factor = 1;
layers(end-2).BiasLearnRateFactor = 20;
layers(end-2).BiasL2Factor = 0;

opts = trainingOptions('sgdm', ...
    'LearnRateSchedule', 'none',...
    'InitialLearnRate', .0001,... 
    'MaxEpochs', 20, ...
    'MiniBatchSize', 128);

convnet = trainNetwork(imds, layers, opts);

rootFolder = 'cifar10Test';
testDS = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
testDS.ReadFcn = @readFunctionTrain;

[labels,err_test] = classify(convnet, testDS, 'MiniBatchSize', 64);

confMat = confusionmat(testDS.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))

% This function simply resizes the images to fit in AlexNet
% Copyright 2017 The MathWorks, Inc.


function I = readFunctionTrain(filename)
% Resize the images to the size required by the network.
I = imread(filename);
I = imresize(I, [227 227]);
end
