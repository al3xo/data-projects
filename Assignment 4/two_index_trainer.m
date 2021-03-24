function [U,S,V,threshold,w,sortb,sorta, accuracy] = two_index_trainer2(data, labels, a, b, feature)
    
    ind_ab = find(labels == a | labels == b);
    data_ab = data(:, ind_ab);
    labels_ab = labels(ind_ab);
    
    na = length(find(labels == a));
    nb = length(find(labels == b));
    
    [U,S,V] = svd(data_ab,'econ'); 
    
    digits = S*V';
    U = U(:,1:feature);
    
    digits_b = digits(1:feature,find(labels_ab == b));
    digits_a = digits(1:feature,find(labels_ab == a));
    
    mb = mean(digits_b,2);
    ma = mean(digits_a,2);
    
    Sw = 0;
    for k = 1:nb
        Sw = Sw + (digits_b(:,k)-mb)*(digits_b(:,k)-mb)';
    end
    for k = 1:na
        Sw = Sw + (digits_a(:,k)-ma)*(digits_a(:,k)-ma)';
    end
    Sb = (mb-ma)*(mb-ma)';
    
    [V2,D] = eig(Sb,Sw);
    [~, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    vb = w'*digits_b;
    va = w'*digits_a;
    
    if mean(vb)>mean(va) % so vb is always less than va
        w = -w;
        vb = -vb;
        va = -va;
    end
    
    sortb = sort(vb);
    sorta = sort(va);
    
    t1 = length(sortb);
    t2 = 1;
    
    while sortb(t1)>sorta(t2)
        t1 = t1-1;
        t2 = t2+1;
    end
    threshold = (sortb(t1)+sorta(t2))/2;

    accuracy = 1 - t2*2 / (na + nb);
end

