function Vocab = vocab(Corpus_path)
%Build a vocabulary with word frequencies for an entire corpus.
%Returns a dictionary `w -> (i, f)`, mapping word strings to pairs of
%word ID and word corpus frequency.

vocab_dictionary = py.collections.Counter();

fid = fopen(Corpus_path);
Corpus = fgets(fid);

while ischar(Corpus)
    %disp(tline)
    tokens = strsplit(Corpus);
    vocab_dictionary.update(tokens);
    Corpus = fgets(fid); 

end


Vocab = py.dict();

%for n = vocab_dictionary.keys()
    %disp(n[0]);
%end
words = keys(vocab_dictionary);
index = 1;
for word = words
    
    val = vocab_dictionary{word{1}};
    tuple = py.tuple({index,val});
    Vocab{word{1}} = tuple;
    index = index+1; 
end

end
