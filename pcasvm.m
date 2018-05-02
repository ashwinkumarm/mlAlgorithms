pca2Svm()
warning off;

function data_set=pcaTest( test,V,xmean,xvar)
x_data=test(:,1:57);
y_data=test(:,58);
y_data(find(y_data == 0)) = -1;
A=(x_data - xmean)./xvar;
temp=A*V;
data_set=[temp y_data];
end


function [V,data_set,xmean,xvar] = pcaTrain(train,k)
x_train = train(:,1:57);
y_train = train(:,58);
y_train(find(y_train == 0)) = -1;
xmean = mean(x_train);
xvar = std(x_train - mean(x_train));
B = (x_train - mean(x_train))./xvar;
covariance=B'*B;
[V,valI] = eigs(covariance,k);  
if (k == 6)
    disp(diag(valI));
end    
temp=x_train*V;
data_set=[temp y_train];
end

function acc = svm(train, test, C)
m = size(train,2);
n = size(test,2);
warning off;
model = fitcsvm(train(:,1:m-1),train(:,m),'boxconstraint', C);
svm_results=predict(model,test(:,1:n-1));

correct=0.0;
for i = 1:length(svm_results)
    if(svm_results(i) == test(i,n))
        correct=correct+1.0;
    end
    
end
acc = 100.0*correct/length(svm_results);
end

function [w b] = findWandB(c,X,Y)
    n = size(X,1);
    m = size(X,2);
    H = diag([ones(1, m), zeros(1, n + 1)]);
    f = [zeros(1,m+1) c*ones(1,n)]';
    p = diag(Y) * X;
    A = -[p Y eye(n)];
    B = -ones(n,1);
    lb = [-inf * ones(m+1,1) ;zeros(n,1)];
    z = quadprog(H,f,A,B,[],[],lb);
    w = z(1:m,:);
    b = z(m+1:m+1,:);
    eps = z(m+2:m+n+1,:);
end   

function acc = accuracy1(X,Y,w,b)
    pred = X * w + b;
    guess = sign(pred);
    %disp([pred Y]);
    r = guess == sign(Y);
    rac = sum(r(:) == 1);
    acc = rac/size(X,1);
end



function pca2Svm()
trainData=load('D:\semester_3\machine learning\Assgn\Assgn4\spam_train.data');
validData=load('D:\semester_3\machine learning\Assgn\Assgn4\spam_validation.data');
testData=load('D:\semester_3\machine learning\Assgn\Assgn4\spam_test.data');


for k = [1,2,3,4,5,6]
   [V,train,m,v] = pcaTrain(trainData,k);
   valid = pcaTest(validData, V, m,v);
   for C = [1,10,100,1000]
      [w,b] = findWandB(C,train(:,1:k), train(:,k+1));
      disp(w);
      disp(b);
      acc = accuracy1(valid(:,1:k),valid(:,k+1),w,b);
      %acc = svm(train,valid,C);
      disp("For k "+ k +" and C "+ C)
      disp(acc);
   end
end   
% test data set with pca
k = 4;
C = 1;
[V,train,m,v] = pcaTrain(trainData,k);
test = pcaTest(testData, V, m,v);
[w,b] = findWandB(C,train(:,1:57), train(:,58));
      acc = accuracy1(test(:,1:57),test(:,58),w,b);
      disp("C "+ C);
      disp(acc);

% test data set with out pca
[w,b] = findWandB(C,trainData(:,1:57), trainData(:,58));
      acc = svm(trainData,testData,C);
      disp("C "+ C);
      disp(acc);

end
