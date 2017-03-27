function words = most_similar_word(vectors,vocab,id2word,word,n)

    dists = py.list();
    words = py.list();
    
    word_id = vocab{word}{1};
    word_vector = vectors(word_id,:);
    
    temp = length(keys(vocab));
    
    for index = 1:temp
         if index ~= word_id
            dot_product = dot(word_vector,vectors(index,:));
            dists.append(dot_product);
         end
    end    
    
    dists = py.numpy.asarray(dists);
    top_ids = py.list(py.numpy.argsort(dists));
    top_ids.reverse();
    top_word = py.list();
    
    for i = 1:n
        id = top_ids(i);
        if id{1} == 0
            continue;
        end
        
        words.append(id2word{id{1}});
    end
    
end
