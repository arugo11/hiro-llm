#let project-document(
  title: none,
  author: none,
  date: none,
  body,
) = {
  set document(title: title, author: author)
  set page(
    paper: "a4",
    margin: (x: 25mm, y: 25mm),
    numbering: "1",
    number-align: center + bottom,
  )
  set text(
    font: ("Libertinus Serif", "Noto Serif CJK JP"),
    lang: "ja",
    size: 10.5pt,
  )
  set par(justify: true, leading: 0.75em)
  set heading(numbering: "1.1")
  show heading: set text(
    font: "Noto Sans CJK JP",
  )

  align(center)[
    #text(size: 20pt, weight: "bold", title)
    #v(1em)
    #text(size: 11pt, author)
    #v(0.5em)
    #text(size: 10pt, date)
  ]

  v(2em)
  body
}
