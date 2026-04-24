# Qualitative Examples

## Strong Retrieval
Query: 75 | Portrait of a Member of the Haarlem Civic Guard | Hals, Frans
First relevant rank: 1 | Precision@5: 1.000 | AP@10: 1.000
Comment: Relevant matches appear immediately, and the top of the list stays inside a tight artist cluster.
Top-5 retrieved:
- 1. 78 | Portrait of a Young Man | Hals, Frans | score=0.9990 | relevant
- 2. 1167 | Portrait of a Man | Hals, Frans | score=0.9175 | relevant
- 3. 571 | A Young Man in a Large Hat | Hals, Frans | score=0.8363 | relevant
- 4. 77 | Adriaen van Ostade | Hals, Frans | score=0.8245 | relevant
- 5. 1168 | Portrait of a Gentleman | Hals, Frans | score=0.7968 | relevant

## Typical Retrieval
Query: 13 | Madonna and Child with Donor | Lippo Memmi
First relevant rank: 8 | Precision@5: 0.000 | AP@10: 0.125
Comment: The shortlist contains meaningful matches, but it also mixes in looser style-level neighbors.
Top-5 retrieved:
- 1. 892 | Madonna and Child, with the Blessing Christ [middle panel] | Lorenzetti, Pietro | score=0.8063 | non-relevant
- 2. 33 | Christ Blessing | Grifo di Tancredi | score=0.7864 | non-relevant
- 3. 417 | Madonna and Child with Saint Jerome, Saint Bernardino, and Angels | Sano di Pietro | score=0.7678 | non-relevant
- 4. 893 | Saint Catherine of Alexandria, with an Angel [right panel] | Lorenzetti, Pietro | score=0.7526 | non-relevant
- 5. 397 | Madonna and Child | Giotto | score=0.7509 | non-relevant

## Hard Case
Query: 5 | The Madonna of Humility | Angelico, Fra
First relevant rank: None | Precision@5: 0.000 | AP@10: 0.000
Comment: This query is difficult: retrieval leans more on broad composition and style than on artist identity.
Top-5 retrieved:
- 1. 396 | Madonna and Child Enthroned | Gentile da Fabriano | score=0.8137 | non-relevant
- 2. 417 | Madonna and Child with Saint Jerome, Saint Bernardino, and Angels | Sano di Pietro | score=0.7500 | non-relevant
- 3. 33 | Christ Blessing | Grifo di Tancredi | score=0.7161 | non-relevant
- 4. 397 | Madonna and Child | Giotto | score=0.6913 | non-relevant
- 5. 344 | The Coronation of the Virgin with Six Angels | Gaddi, Agnolo | score=0.6814 | non-relevant
