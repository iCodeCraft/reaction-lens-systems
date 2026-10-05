# Third-party materials

The MIT license applies to the author's software and learned Reaction Lens head.
The authored manuscript and original figures use [CC BY 4.0](paper/LICENSE).
Third-party licenses are separate:

| Material | Source and terms | Use here |
| --- | --- | --- |
| SPECTER2 base | [Model card](https://huggingface.co/allenai/specter2_base), Apache-2.0 | Frozen encoder; downloaded from its pinned upstream snapshot |
| SPECTER2 proximity adapter | [Upstream model](https://huggingface.co/allenai/specter2), Apache-2.0 | Downloaded separately; retain upstream files and terms |
| Reactome data and derived descriptions | [Reactome license](https://reactome.org/license), CC0 | Version-97 structured descriptions and recorded association metadata |
| Publication text | Original publications and their providers | Scientific input; no blanket relicensing of third-party abstracts |
| Python dependencies | Individual package licenses | Resolved versions recorded in uv.lock |

The evaluation pack contains cached numeric representations, PMID identifiers
and recorded reaction links. It does not distribute the training corpus or
upstream encoder weights.

License sources checked on 5 October 2026. Reactome art/branding uses different
terms from its data; no Reactome artwork is claimed as original project art.

## Recorded latency inputs

`examples/latency-examples.json` preserves the exact five measured title/abstract
inputs and their original diagnostic metadata so its recorded hash remains
verifiable. The full-catalog benchmark reads only `pmid`, `title` and `abstract`;
the retained historical menu fields do not restrict its candidates.

The source publications are identified by PubMed IDs
[10022914](https://pubmed.ncbi.nlm.nih.gov/10022914/),
[10047984](https://pubmed.ncbi.nlm.nih.gov/10047984/),
[10049780](https://pubmed.ncbi.nlm.nih.gov/10049780/),
[10067896](https://pubmed.ncbi.nlm.nih.gov/10067896/) and
[10069812](https://pubmed.ncbi.nlm.nih.gov/10069812/).
These third-party abstracts are not covered by this project's MIT or manuscript
license. PubMed explicitly attributes copyright for PMID 10049780 to Academic
Press. Redistribution permission has not been established for all five
abstracts. Public availability
in PubMed alone is not a grant of an open license (see
[NCBI policies](https://www.ncbi.nlm.nih.gov/home/about/policies/)).
