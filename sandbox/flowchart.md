```mermaid
flowchart TD
    Start --> Q1
    Q1[Can we synthesize variants for the initial training set?] --> |Yes| Q2
    Q1 --> |No| Q3

    Q2[Is it epistatic or not?] --> |Epistatic| Q4
    Q2 --> |Not epistatic| MLDE

    Q4[Is it binding or enzymatic activity?] --> |Binding| Q5
    Q4 --> |Enzymatic activity| Q6

    Q5[Can we screen 384 variants?] --> |Yes| ESM-IF_ftMLDE
    Q5 --> |No| Triad_ftMLDE

    Q6[Can we screen 384 variants?] --> |Yes| EVmutation_ftMLDE
    Q6 --> |No| Edit_distance_+_ESM-IF_ftMLDE

    Q3[Is it epistatic or not?] --> |Epistatic| Edit_distance_ftMLDE
    Q3 --> |Not epistatic| DE_MLDE
