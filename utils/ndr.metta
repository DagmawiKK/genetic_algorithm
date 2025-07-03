!(import! &self rand)

(= (car $x) (car-atom $x))
(= (cdr $x) (cdr-atom $x))
(= (cons $x $xs) (cons-atom $x $xs))

(= (zip $xs $ys)
   (if (or (== $xs ()) (== $ys ()))
       ()
       (cons (cons (car $xs) (cons (car $ys) ()))
             (zip (cdr $xs) (cdr $ys)))))
(= (roulette-selection-space $population $fitnesses)
   (let* (
          ($w_max (max-atom $fitnesses))
          ($zipped (zip $fitnesses $population))  
         )
     (if (<= $w_max 0)
         (superpose $population) 
         (let* (
                ($pair (superpose $zipped))  
                ($fit (car $pair))           
                ($ind (car (cdr $pair)))    
                ($rand (random-float 0 1))
                ($threshold (/ $fit $w_max))
               )
             (if (> $threshold $rand)
                 $ind
                 () 
             )
         )
     )
)
)



!(collapse-bind (roulette-selection-space ((A) (B) (C) (D)) (0.1 0.4 0.3 0.2)))
