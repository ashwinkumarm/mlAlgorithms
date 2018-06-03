runNGuassian();

function runNGuassian()
[trainData,testData] = loadData();
[o, prior, m,s] = meanAndVar(trainData);
a = accuracy(testData,o,prior, m,s);
disp(a);
end

function [trainData, testData] = loadData()
trainData=load('D:\semester_3\machine learning\Assgn\Assgn4\spam_train.data');
testData=load('D:\semester_3\machine learning\Assgn\Assgn4\spam_test.data');
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
acc = a/size(testData,1);
end