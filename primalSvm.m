%Load data
inp = importdata('D:\semester_3\machine learning\Assgn\Assgn1\mystery.data');
%Class Label
Y = inp(:,5);
%Input - Using cubic kernel  
X = [inp(:,1).*inp(:,1).*inp(:,1) inp(:,2).*inp(:,2).*inp(:,2) inp(:,3).*inp(:,3).*inp(:,3) inp(:,4).*inp(:,4).*inp(:,4)];
H = eye(size(X,2)+1);
H(size(X,2)+1,size(X,2)+1) = 0;
f = zeros(size(X,2)+1,1);
Z = [X ones(1000,1)];
A = -diag(Y) * Z ;
c = -1 * ones(1000,1);
w = quadprog(H,f,A,c);
disp(w);
b = 0;
q = (X*w(1:4,1) + w(5,1));
disp(w(5,1));
bo = diag(Y) * q ;
[row, col] = find(abs(bo - 1.000) < 0.1);
disp(X(row,:));
