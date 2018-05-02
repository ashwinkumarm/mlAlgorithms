start();

function start()
[trainFile, XTrain,YTrain] = loadTrainFile();
[testFile, XTest,YTest] = loadTestFile();
EV = eigenValues(XTrain);
sampleData(EV, trainFile,testFile);
end


function [trainFile, X,Y] = loadTrainFile()
trainFile=load('D:\semester_3\machine learning\Assgn\Assgn4\spam_train.data');
X = trainFile(:,1:57);
Y = trainFile(:,58);
end

function [testFile, X,Y] = loadTestFile()
testFile=load('D:\semester_3\machine learning\Assgn\Assgn4\spam_test.data');
X = testFile(:,1:57);
Y = testFile(:,58);
end

function EV = eigenValues(x_train)
B = (x_train - repmat(mean(x_train), size(x_train,1), 1))./ repmat(std(x_train), size(x_train,1), 1);
covariance=B'*B;
[EV,val] = eigs(covariance,10);
end

function sampleData(EV, trainData, testData)
for k = [1,2,3,4,5,6,7,8,9,10]
V=EV(:,1:k);
[a,b]=size(V);
piF=zeros(a,1);
for i =1:a
    total=0;
    for j=1:b
        total=total+(V(i,j)^2);
    end
    piF(i,1)=total/k;
end
for s = 1:20
    acc = 0;
for iter = 1 : 100
samIndex = unique(randsample(57,s,true,piF ));
train_data = construct(samIndex, trainData);
test_data = construct(samIndex, testData);
[o,pr,m,st] = meanAndVar(train_data);
a = accuracy(test_data,o,pr,m,st);
acc = acc + a;
end
accAvg = (acc/100);
loss = 100 - accAvg;
disp("K " + k + " S " + s+ " Accuracy is "+ accAvg+ " loss is " + loss);
end
end
end

function d = construct(ind, data)
d = data(:,[ind',58]);
end

function [o,prior,  m,s] = meanAndVar(trainData)
o = unique(trainData(:,size(trainData,2)));
for k = 1: length(o)
    prior(k) = sum(trainData(:,size(trainData,2)) == o(k))/ size(trainData,1);
    for i = 1 : size(trainData,2)-1
        x = [];  pos = 1;
        for tj = 1: size(trainData,1)
            if trainData(tj,size(trainData,2)) == o(k)
            x(pos) = trainData(tj,i);
            pos = pos +1;
            end    
        end  
        m(k,i) = mean(x);
        s(k,i) = std(x);
    end    
end    
end

function acc = accuracy(testData,o, prior, m,s)
index = [];
for it  = 1 : size(testData,1)
    for kt = 1:length(o)
        p = 1;
        for ft = 1: size(testData,2)-1
            n = normpdf(testData(it,ft),m(kt,ft),s(kt,ft));
            n = prod(n);
            if( isnan(n) == 0)
                p = p * n;
            end    
            if n == 0
                p = p * 0.00000000000000000000001;
            end    
        end    
       finP(kt) = (prior(kt) * p);
    end
    [v,ind] = max(finP);
    index(it) = ind;
end    
a = sum(index' == testData(:,size(testData,2)) +1);
acc = a/size(testData,1)* 100;
end


