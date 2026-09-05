# Writing guide for blog posts

How posts on this site should read. Compiled 2026-09-05 from Wikipedia's "Signs of AI writing"
(WikiProject AI Cleanup), the Pangram and Olivia Cal pattern guides, the Technically Product
"How not to write like ChatGPT" piece, Nielsen Norman Group's F-pattern research, and a read of
the ten posts that were rewritten that day.

The short version: write the way you would explain the project to a colleague at a desk.
Say what happened, what it cost, what broke, and what you would do differently. Stop there.

## Titles

A title says what the post is about. It should work as a line in a table of contents.

Good: "Vandubbi: a wireless underwater ROV built from PVC pipe", "My homelab setup",
"A year of building a 3D bioprinter".

Avoid:
- Aphorisms and slogans ("Hardware is hard").
- Poetic or nostalgic phrasing ("The flying years").
- Suspense or scale for effect ("...half a million machines I will never see").
- The "what X taught me about Y" formula.
- "X: the Y of Z" and other metaphor titles.

## Headings

Same rule. A heading names what the section contains, in plain words: "Waterproofing",
"Cost", "Buoyancy and trim", "Backups". Avoid clever headings that personify a system, set
up a twist, or use parallel wordplay ("The failsafe involves no code", "Verifiers beat judges",
"The database that died"). Sentence case, no title case.

## Sentences

- Most sentences under 20 words. Let a few run long when the idea needs it. Vary the length.
- Use contractions. Start sentences with "And" or "But" if that is how you would say it.
- Say the thing. Skip the sentence that first says what it isn't ("X, not Y"; "not just X, it's Y";
  "X doesn't do Y; it does Z"; "X rather than Y"). State the claim and move on.
- No em dashes or en dashes. Use a comma, a period, or parentheses.
- Don't end paragraphs on a quotable line. If a paragraph ends with something that sounds like
  a fortune cookie ("Memory is not an instrument", "The suite is the asset; the model is a tenant"),
  cut it or replace it with the concrete fact it was standing in for.
- Don't personify hardware or physics ("the ink negotiates", "gravity has opinions",
  "the sky invoices every mistake"). Say what the material or the system does.
- Drop the "which is ..." tag clause that adds a reflection to the end of a factual sentence.
- Watch for tic words: honest/honestly, actually, the whole X, the actual X, earns its keep,
  load-bearing, the real question, at its core, deliberate/on purpose (when the reason is
  already given), boring on purpose.
- No colon reveals ("The trick that fixes this: ..."). Write the sentence.
- Metaphor budget: one per post, if any. Tuition, diploma, liturgy, religion, syllabus,
  ledger, confession, and the like are the first things to cut.

## Structure

- First paragraph says what the post is about and what the reader gets.
- One idea per paragraph. Two to five sentences is normal. Uneven is fine.
- Captions under images describe the image. "This is the frame during a December test."
  Not "That is what an ROV lab looks like when the lab is a balcony."
- The closing numbered list (period posts) is a list of specific things learned, each a plain
  sentence or two with a number, a part name, or a concrete event in it. Not an aphorism
  followed by its explanation.
- End when the content ends. No summary paragraph, no upbeat send-off.

## Technical depth

Aim for a strong undergraduate or master's write-up, the level of a good project report.

- One formula per concept, with the numbers plugged in once. Don't stack derivations.
- Name a method (Poisson reconstruction, Cohen's kappa, a change detector) and say what it's for.
  Don't reproduce its equation unless the post is about that method.
- Prefer a rule of thumb with its number ("about 2.5 dB per centimetre of water at 2.4 GHz")
  over the constant it was derived from.
- Cut specialist vocabulary a final-year student wouldn't know (Rabinowitsch correction, Wilson
  interval, COSE signature, JUMBF box) unless it's the point of the section.
- Code blocks stay short and readable. A 20-line contract is fine. A fully general library is not.
- The owner said this on 2026-09-05 after a first pass came out "PhD level".

## Keep

- Specific numbers at their real precision (₹20,372, 891,105 points, 0.7366 AUC).
- Filenames, part numbers, dates, issue numbers.
- Opinions and uncertainty in first person ("I don't know why this works", "I'd skip this").
- Jokes that a person would say out loud. One or two per post.

## Don't

- Invent a number, quote, date, or source to sound concrete. If it's unknown, say so or leave it out.
- Add "honest" to make a sentence sound candid.
