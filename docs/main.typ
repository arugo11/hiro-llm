#import "template.typ": project-document

#show: project-document.with(
  title: "hiro-llm 技術文書",
  author: "著者名",
  date: datetime.today().display("[year]年[month]月[day]日"),
)

= はじめに

このファイルを起点として、Typst で文書を執筆できます。
日本語には Noto Serif CJK JP、欧文には Libertinus Serif を使用します。

== 数式とコード

インライン数式は $y = f(x)$ のように記述できます。
別行立ての数式も利用できます。

$
  "loss" = - sum_i p_i log q_i
$

コードは次のように記述します。

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```
