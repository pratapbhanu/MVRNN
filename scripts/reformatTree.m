function [inc numnode newt] = reformatTree(thisNode, t, upnext)

% binarize

kids = t.kids(:,thisNode);
kids = kids(find(kids));
kkk = t.isLeafnode(kids(1));

while length(kids) == 1 && kkk ~= 1
    kkids = t.kids(:,kids(1));
    kkids = kkids(find(kkids));
    
    t.pp(kids(1)) = -1;
    t.pp(kkids) = thisNode;
    t.kids(1:length(kkids),thisNode) = kkids;
    
    kids = kkids;
    kkk = t.isLeafnode(kids(1));
end

numnode = 0;
kkk = t.isLeafnode(kids(1));
if length(kids) == 1 && kkk
    t.isLeafnode(thisNode) = 1;
    t.pp(kids(1)) = -1;
    t.kids(:,thisNode) = 0;
    inc = 0;
    numnode = 1;    
else
    inc = 0;

    for k = 1:length(kids);
        kkk = t.isLeafnode(kids(k));
        if ~kkk
            [thisinc thisnumnode newt] = reformatTree(kids(k), t, upnext+inc);
            inc = inc+ thisinc;
            t = newt;
            numnode = numnode+thisnumnode;
        else
            numnode = numnode+1;
        end
    end

    next = upnext + inc;
    n = length(kids);
    last = kids(end);
    start = n-1;
    while n >= 2
        if (n == 2)
            next = thisNode;
        else
            next = next + 1;
            inc = inc+1;
        end
        
        t.pp(last) = next;
        t.pp(kids(start)) = next;
        
        t.kids(:, next) = 0;
        t.kids(1, next) = kids(start);
        t.kids(2, next) = last;


        last = next;
        start = start-1;
        n = n - 1;
    end
end

newt = t;