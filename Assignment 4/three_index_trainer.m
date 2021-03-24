function [U,S,V,threshold,w,sortb,sorta] = three_index_trainer(data, labels, a, b, c, feature)
       
    ind_abc = find(labels == b | labels == a | labels == c);
    data_abc = data(:, ind_abc);
    labels_abc = labels(ind_abc);

    na = length(find(labels == a));
    nb = length(find(labels == b));
    nc = length(find(labels == c));

    [U,S,V] = svd(data_abc,'econ'); 
    digits = S*V';
    U = U(:,1:feature); 

    digits_a = digits(1:feature, find(labels_abc == a));
    digits_b = digits(1:feature, find(labels_abc == b));
    digits_c = digits(1:feature, find(labels_abc == c));

    ma = mean(digits_a,2);
    mb = mean(digits_b,2);
    mc = mean(digits_c,2);

    m_all = mean(digits);

    % scatter matrices
    Sw = 0;
    for i = 1:na
        Sw = Sw + (digits_a(:, i) - ma)*(digits_a(:, i) - ma)';
    end
    for i = 1:nb
        Sw = Sw + (digits_b(:, i) - mb)*(digits_b(:, i) - mb)';
    end
    for i = 1:nc
        Sw = Sw + (digits_c(:, i) - mc)*(digits_c(:, i) - mc)';
    end

    Sb = (ma-m_all)*(ma-m_all)'  + (mb-m_all)*(mb-m_all)'  + (mc-m_all)*(mc-m_all)';

    % eigenvalues
    [V2,D] = eig(Sb,Sw);
    [~, ind] = max(abs(diag(D)));

    temp = [abs(diag(D)) (1:length(abs(diag(D))))'];
    temp = sortrows(temp, 'descend');
    w1 = V2(:, temp(1, 2));
    w2 = V2(:, temp(2, 2));
    w3 = V2(:, temp(3, 2));

    v1_a = w1'*digits_a;
    v2_a = w2'*digits_a;
    v3_a = w3'*digits_a;

    v1_b = w1'*digits_b;
    v2_b = w2'*digits_b;
    v3_b = w3'*digits_b;

    v1_c = w1'*digits_c;
    v2_c = w2'*digits_c;
    v3_c = w3'*digits_c;

    % sorta_1 = sort(v1_a);
    % sorta_2 = sort(v2_a);
    % sortb_1 = sort(v1_b);
    % sortb_2 = sort(v2_b);
    % sortc_1 = sort(v1_c);
    % sortc_2 = sort(v2_c);
    % 
    % find 2 cutoff values for each dim
    % 
    % dim 1
    % v1_means = [mean(v1_a) mean(v1_b) mean(v1_c)]; 
    % [v1_means, I] = sort(v1_means);
    % 
    % t1 = (v1_means(1) + v1_means(2)) / 2;
    % t2 = (v1_means(2) + v1_means(3)) / 2;
    % 
    % 
    % 
    % %
    % t1 = length(sortb);
    % t2 = 1;
    % 
    % while sortb(t1)>sorta(t2)
    %     t1 = t1-1;
    %     t2 = t2+1;
    % end
    % threshold = (sortb(t1)+sorta(t2))/2;

end

