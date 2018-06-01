function primalSvmWithSlack()
    [X,Y] = primalSvmTrainData();
    c = primalSvmValidData(X,Y);
    primalSvmTestData(c,X,Y);
end

function [X,Y] = primalSvmTrainData()
  inp = importdata('D:\semester_3\machine learning\Assgn\Assgn2\wdbc_train.data');
  Y = inp(:,1);
  X = inp(:,2:11);
  disp("----------Training Data-----------");
  runForDiffC(X,Y); 
end

function c = primalSvmValidData(Xinp, Yinp)
  inp = importdata('D:\semester_3\machine learning\Assgn\Assgn2\wdbc_valid.data');
  Y = inp(:,1);
  X = inp(:,2:11);
  disp("----------validation Data-----------");
  c = runForDiffCValid(Xinp,Yinp, X,Y);
end

function primalSvmTestData(c,Xinp,Yinp)
  inp = importdata('D:\semester_3\machine learning\Assgn\Assgn2\wdbc_test.data');
  Y = inp(:,1);
  X = inp(:,2:11);
  disp("----------Test Data-----------");
  [w , b] = findWandB(c,Xinp,Yinp);
  a = accuracy(X,Y,w,b);
  disp('for c value: ')
  disp(c);
  disp('accuracy is')
  disp(a);
end

function C = runForDiffC(X,Y)
    maxAcc = -1;
    for c = [1,10,10^2,10^3,10^4,10^5,10^6,10^7,10^8]
      [w , b] = findWandB(c,X,Y);
      a = accuracy(X,Y,w,b);
      disp('for c value: ')
      disp(c);
      disp('accuracy is')
      disp(a);
      if maxAcc < a
        maxAcc = a;
        C = c;
      end  
    end    
end

function C = runForDiffCValid(X,Y,Xo,Yo)
    maxAcc = -1;
    for c = [1,10,10^2,10^3,10^4,10^5,10^6,10^7,10^8]
      [w , b] = findWandB(c,X,Y);
      a = accuracy(Xo,Yo,w,b);
      disp('for c value: ')
      disp(c);
      disp('accuracy is')
      disp(a);
      if maxAcc < a
        maxAcc = a;
        C = c;
      end  
    end    
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


function acc = accuracy(X,Y,w,b)
    pred = X * w + b;
    guess = sign(pred);
    r = guess == sign(Y);
    rac = sum(r(:) == 1);
    acc = rac/size(X,1);
end