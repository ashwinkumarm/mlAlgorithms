function dualSvm()
[X,Y] = dualSvmTrain();
[c,sigma] = dualSvmValid(X,Y);
dualSvmTest(X,Y,c,sigma);
end

function [X,Y] = dualSvmTrain()
inp = importdata('D:\semester_3\machine learning\Assgn\Assgn2\wdbc_train.data');
Y = inp(:,1);
X = inp(:,2:11);
disp('------------------Training data ---------------');
[c,sigma] = diffKandSigma(X,Y);
end

function [c,sigma] = dualSvmValid(Xinp, Yinp)
inp = importdata('D:\semester_3\machine learning\Assgn\Assgn2\wdbc_valid.data');
Y = inp(:,1);
X = inp(:,2:11);
disp('------------------Valid test data ---------------');
[c,sigma] = diffKandSigmaValid(Xinp, Yinp, X,Y);
end

function dualSvmTest(Xinp, Yinp, c, sigma)
inp = importdata('D:\semester_3\machine learning\Assgn\Assgn2\wdbc_test.data');
Y = inp(:,1);
X = inp(:,2:11);
disp('------------------ Test data ---------------');
lambda = findlambda(Xinp,Yinp,c,sigma);
b = findB(lambda, Xinp, Yinp, c,sigma);
acc = accuracy(lambda,X,Y,sigma,b);
disp('values of c and sigma are');
disp(c);
disp(sigma);
disp('accuracy is');
disp(acc);
end

function [C,Sigma] = diffKandSigmaValid(Xinp, Yinp, X,Y)
maxAcc = -1;
for c = [1,10,10^2,10^3,10^4,10^5,10^6,10^7,10^8]
    for sigma = [0.1,1,10,100,1000]
        lambda = findlambda(Xinp,Yinp,c,sigma);
        b = findB(lambda, Xinp, Yinp, c,sigma);
        acc = accuracy(lambda,X,Y,sigma,b);
        disp('values of c and sigma are');
        disp(c);
        disp(sigma);
        disp('accuracy is');
        disp(acc);
        if(maxAcc < acc)
            maxAcc = acc;
            C = c;
            Sigma = sigma;
        end
    end
end
end

function [C,Sigma] = diffKandSigma(X,Y)
maxAcc = -1;
for c = [1,10,10^2,10^3,10^4,10^5,10^6,10^7,10^8]
    for sigma = [0.1,1,10,100,1000]
        lambda = findlambda(X,Y,c,sigma);
        b = findB(lambda, X, Y, c,sigma);
        acc = accuracy(lambda,X,Y,sigma,b);
        disp('values of c and sigma are');
        disp(c);
        disp(sigma);
        disp('accuracy is');
        disp(acc);
        if(maxAcc < acc)
            maxAcc = acc;
            C = c;
            Sigma = sigma;
        end
    end
end
end


function lambda = findlambda(X,Y,c,sigma)
n = size(X,1);
for i = 1:n
    for j = 1:n
        H(i,j) = Y(i) * Y(j) *  guassianKernel(X(i,:), X(j,:), sigma);
    end
end
f = -1 * ones(1,n);
A = zeros(1,n);
B = 0;
Aeq = Y';
Beq = 0;
ub = [c * ones(n,1)];
lb = zeros(n,1);
[lambda] = quadprog(H,f,A,B,Aeq,Beq,lb,ub);
end


function b = findB(lambda, X, Y, c, sigma)
b = 0;
noOfSv = 0;
for i = 1:size(X,1)
    b = 1/Y(i);
    for j = 1:size(X,1)
        if lambda(j) > 0.00001  && lambda(j) < c
            b = b - lambda(j) * Y(j) * guassianKernel(X(i,:),X(j,:),sigma);
            noOfSv = noOfSv + 1;
        end
    end
end
b = b / noOfSv;
end

function acc = accuracy(lambda, X, Y,sigma,b)
n = size(X,1);
guess = zeros(size(X,1));
for i = 1:n
    g = 0;
    for j = 1:n
        g = g + lambda(j) * Y(j) * guassianKernel(X(i,:),X(j,:),sigma);
    end
    guess(i) = g+b;
end
guess = sign(guess);
r = guess == sign(Y);
rac = sum(r(:) == 1);
acc = rac/size(X,1);
end

function k = guassianKernel(x, z, sigma)
nr = (x-z);
nr = - (nr * nr');
k = exp(nr/(2 * sigma * sigma));
end