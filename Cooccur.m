function cooccurrences = Cooccur(vocab,Corpus_path,window_size)

fid = fopen(Corpus_path);
Corpus = fgets(fid);

vocab_len = length(keys(vocab));
cooccurrences = sparse(vocab_len,vocab_len);
index = 1;

while ischar(Corpus)
    tokens = strsplit(Corpus);
    token_ids = py.list();
    for token = tokens
         
        token_ids.append(vocab{token{1}}{1});
    end %tokens
    
    center_i = 1;
    for center_id = token_ids
        
        context_ids = token_ids(max(1 , center_i - window_size):center_i);
        contexts_len = length(context_ids);
        left_i = 1;
        
        for left_id = context_ids
            
            %Distance from center word
            distance = contexts_len - (left_i-1);
            %# Weight by inverse of distance between words
            increment =  1.0 / typecast(distance,'double');
            %Build co-occurrence matrix symmetrically (pretend we
            %are calculating right contexts as well)
            cooccurrences(center_id{1}, left_id{1}) =  cooccurrences(center_id{1}, left_id{1}) +  increment;
            cooccurrences(left_id{1}, center_id{1}) =  cooccurrences(left_id{1}, center_id{1}) +  increment;
                   
            left_i = left_i + 1;
            
        end % fill matrix
       
        center_i = center_i + 1;
        
    end%find Context
    
    Corpus = fgets(fid);
end %While loop

end