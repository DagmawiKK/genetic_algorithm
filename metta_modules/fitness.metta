(= (fitness-rec $cand $a $b $i $gene_len)
    (if (>= $i $gene_len)
        0
        (let*
            (
                ($c (car $cand))
                ($a1 (car $a))
                ($b1 (car $b))
                ($maxAB (max-atom ($a1 $b1)))
                ($diff (- $c $maxAB))
                ($emergence (max-atom ($diff 0)))
                ($minAB (min-atom ($a1 $b1)))
                ($contrib (* $minAB $emergence))
                ($rest (fitness-rec (cdr $cand) (cdr $a) (cdr $b) (+ $i 1) $gene_len))
            )
            (+ $contrib $rest)
        )
    )
)

(= (fitness $individual $input_a $input_b $gene_len)
    (let*
        (
            ($total (fitness-rec $individual $input_a $input_b 0 $gene_len))
            ($normalized (/ $total $gene_len))
            ($clamped (min-atom ($normalized 1.0)))
        )
        $clamped
    )
)
