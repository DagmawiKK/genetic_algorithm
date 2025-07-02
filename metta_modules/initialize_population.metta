
(= (randfloat-array $gene_len)
   (if (<= $gene_len 0)
       ()
       (cons (random-float &rng 0 1) 
             (randfloat-array (- $gene_len 1)))))


(= (make-pop-rec $i $pop_size $gene_len)
   (if (>= $i $pop_size)
       ()
       (cons (randfloat-array $gene_len) 
             (make-pop-rec (+ $i 1) $pop_size $gene_len))))

(= (make-pop $pop_size $gene_len)
   (make-pop-rec 0 $pop_size $gene_len))

