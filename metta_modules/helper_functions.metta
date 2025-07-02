; !(import! &self rand)

(= (car $x) (car-atom $x))
(= (cdr $x) (cdr-atom $x))
(= (cons $x $xs) (cons-atom $x $xs))

; max2(a,b) = if a > b then a else b
(= (max2 $a $b)
    (if (> $a $b)
        $a
        $b))

; min2(a,b) = if a < b then a else b
(= (min2 $a $b)
    (if (< $a $b)
        $a
        $b))

; max0(x) = if x > 0 then x else 0
(= (max0 $x)
    (if (> $x 0)
        $x
        0))

; add(a,b) = a + b
(= (add $a $b)
    (+ $a $b))

; sub(a,b) = a - b
(= (sub $a $b)
    (- $a $b))

; mul(a,b) = a * b
(= (mul $a $b)
    (* $a $b))

; div(a,b) = if b == 0 then 0 else a / b
(= (div $a $b)
    (if (== $b 0)
        0
        (/ $a $b)))
