function wirte_to_csv(Vocab,cooccurrences,id2word)

    [i_main,i_context,cooccur] = find(cooccurrences);
    
    words = keys(Vocab);
    disp(words);
    
    data_frame = py.pandas.dataframe();
    
end