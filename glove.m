function glove()
%This function open text file and extract vocab from that
%Take one argument corpus file location

%Glove Parametres
Corpus_path = 'Corpus.txt';
window_size = 10;
vector_size = 100;
iterations = 500;
min_count = 10;
learning_rate = 0.05;
x_max = 100;
alpha = 0.75;
normalized = 1;

Vocab = vocab(Corpus_path);
disp(length(Vocab));


id2word = py.dict();
index = 1;

%Create a dictionary id --> word
for key = keys(Vocab)
    
    id2word{index} = key{1};
    index = index + 1;
    
end
cooccurrences = Cooccur(Vocab,Corpus_path,window_size);
%write_to_csv(Vocab,cooccurrences,id2word);
W = train(Vocab,cooccurrences,vector_size,iterations,learning_rate,x_max,alpha);
Vectors = evaluate(W,Vocab,normalized,vector_size);
words = most_similar_word(Vectors,Vocab,id2word,'graph',15);
disp(words);


end




