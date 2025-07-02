; Define mutation rate constant
!(import! &self rand)


; Recursive helper that walks through index positions
(= (mutate-rec $ind $std $mutation-rate $i $len)
   (if (>= $i $len)
       ()
       (let* (
           ($gene (index-atom $ind $i))
           ($r (random-float &rng 0 1))
           ($mutated
               (if (< $r $mutation-rate)
                   (let* (
                       ($noise (gaussian-random 0 $std))
                       ($added (+ $gene $noise))
                       ($clipped (max-atom (0 (min-atom (1 $added)))))
                   )
                       $clipped
                   )
                   $gene
               )
           )
           ($rest (mutate-rec $ind $std $mutation-rate (+ $i 1) $len))
       )
           (cons-atom $mutated $rest)
       )
   )
)

; Entry function (mutation starts from index 0)
(= (mutate $ind $std $mutation-rate)
    (mutate-rec $ind $std $mutation-rate 0 (size-atom $ind)))

; !(mutate ([0.7812015273974839 0.9304311460475365 0.8856786021199445 0.9385503728389093 0.6386600723962299 0.4198448259642038 0.2754761373239567 0.37416093057638533) 0.5)