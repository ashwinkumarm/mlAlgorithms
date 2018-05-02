inp = readtable('D:\semester_3\machine learning\Assgn\Assgn5\congress.txt', 'TreatAsEmpty',{''},'ReadVariableNames',false);
stdinp = standardizeMissing(inp,'?');
cleaninp = rmmissing(stdinp);
infoGainArray = zeros(17);
for i = 1:17
    for j = 1:17
        if i ~= j
        infoGainArray(i,j) = calcInfoGain(cleaninp, i, j);
        end
    end
end

[ Tree,Cost ] =  UndirectedMaximumSpanningTree(infoGainArray);
disp(Tree);
disp(Cost);
h1 = view(biograph( Tree ))

function gain = calcInfoGain(cleaninp, a, b)
noOfRows = size(cleaninp,1);
aT = unique(cleaninp(:,a));
bT = unique(cleaninp(:,b));
infoGain = 0;
for i = 1:size(aT,1)
    for j = 1:size(bT,1)
        PAB = 0;
        PA = 0;
        PB = 0;
        for m = 1 : size(cleaninp,1)
            if strcmp(cleaninp{m,a},aT{i,1}) 
                PA = PA +1;
            end   
            if strcmp(cleaninp{m,b},bT{j,1})
                PB = PB +1;
            end   
            if strcmp(cleaninp{m,a},aT{i,1}) && strcmp(cleaninp{m,b},bT{j,1})
                PAB = PAB +1;
            end   
        end    
        PAB = PAB/noOfRows;
        PA = PA/noOfRows;
        PB = PB/noOfRows;
        infoGain = infoGain + (PAB * log(PAB/(PA * PB)));
    end   
end
disp(infoGain);
gain = infoGain;
end

