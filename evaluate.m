function vector = evaluate(W,Vocab,normalized,vector_size)

% Now merge main vector with context vector
vocab_size = length(keys(Vocab));


vector = rand(vocab_size,vector_size);

for index = 1:vocab_size
    
    main = W(index,:);
    context = W(index+vocab_size,:);
    save matrices main context;
    X = load('matrices');
    Bigmat = cell2mat(struct2cell(X));
    vector(index,:) = mean(Bigmat);
end

end