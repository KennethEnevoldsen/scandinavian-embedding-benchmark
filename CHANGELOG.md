# CHANGELOG



## v0.13.4 (2024-04-08)

### Documentation

* docs: minor updates to tables ([`9f7da4b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/9f7da4ba0cbdbead8b1d2636b4a7c85860ab4619))

### Fix

* fix: updated ruff dependency ([`7e7a7ae`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7e7a7ae84b51f879b9abd66badfc3f070a651ab1))

* fix: Fixed broken HF links to scandeval ([`b99e300`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b99e300ba4db11fca58b51f2d91b4ce9c5388f57))

### Unknown

* Merge pull request #174 from KennethEnevoldsen/minor-fixes

fix: Fixing links for prs ([`5dc4ca3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/5dc4ca332dacd19e1fa8892f812f31179bdad2d3))


## v0.13.3 (2024-02-23)

### Documentation

* docs: Updated tables to include task subtypes ([`e119c58`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e119c58d5240cc39145b7072fc1bd32d22bb5e2e))

### Fix

* fix: Added task subtypes to tasks

This follows the denotion in the paper. A task can have multiple task subtypes but only one task type. ([`7fc9ed5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7fc9ed52c322d87a12960efa3eca4aab58afaf33))

### Unknown

* Merge pull request #162 from KennethEnevoldsen/add-task-subtypes

Added task subtypes ([`363ab09`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/363ab0980ac4587318821ddf9b056db2ba93ea34))


## v0.13.2 (2024-02-19)

### Fix

* fix: Pass the task for encode_queries, and encode_corpus

This yield notable performance improvements for the instruct models for retrieval tasks ([`9992e80`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/9992e802ec62020806f1e1155a5ab5d9b74265a5))

### Unknown

* Merge pull request #156 from KennethEnevoldsen/fix_instruct_tuned_embed

fix: Pass the task for encode_queries, and encode_corpus ([`13786fe`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/13786fe4aaed4de9e963fcb5a6d959bdedc1c16d))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into fix_instruct_tuned_embed ([`69b2ae2`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/69b2ae208f3481c9a482a4f90afde90ff563f1e3))


## v0.13.1 (2024-02-19)

### Chore

* chore: remove test file ([`14f9935`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/14f9935d5e08d1f0110525e9ffe887d0cecd4c17))

### Documentation

* docs: Updte docs script to handle new name format ([`aa171dd`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/aa171ddbee4033d2e5881ccdbd7bc9c7c0459bcc))

### Fix

* fix: fix incorrect emb. size for e5 large instruct ([`7865ad7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7865ad7523e82b4bc3cfa0d1db3a9338117aec80))

* fix: Added final for mult. e5 instruct, including speed test of ref. system ([`08e1779`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/08e17792985053163d54dd1ab0db8d151a3cd272))

* fix: added multilingual-e5-large-instruct ([`56bfc16`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/56bfc16a4adcae0c4a565d7f1d711d901b4aa931))

* fix: rename model_architecture to architecture to not take up protected attribute for pydantic ([`f845a49`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f845a4995f6360e2b2b2b10870ce07305df14517))

### Unknown

* Merge pull request #155 from KennethEnevoldsen/add-multilingual-instruct

Add multilingual e5 instruct ([`c2cca49`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c2cca4935bae0a5c4fb429d0e8337e18fbf22d33))


## v0.13.0 (2024-02-19)

### Feature

* feat: Ensure that all model names are consistent

i.e. that they have the same name as they would have on the benchmark ([`c2299cd`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c2299cd2a4871ec2b49fbfa1f175dea4d2661f47))

### Fix

* fix: made the to method optional on the encoder ([`157a91c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/157a91c8d4846ab7155e625327d60f44b1f0763a))

* fix: Add to method to lazyloadencoder ([`0b6d0be`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0b6d0bef30c4c7b32a5384cc44b49a0f1c1070eb))

* fix: Ensure return type is always np.ndarray ([`e8d3994`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e8d3994604503ac168d334c2878b179aae52b739))

* fix: Ensure return type is always np.ndarray ([`06c5cd8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/06c5cd839735bc585145ac9a7b66626e510f512f))

### Unknown

* Merge pull request #153 from KennethEnevoldsen/ensure-consistent-names

Ensure consistent names ([`83fd962`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/83fd962a0e84692052b6bdff9aa1e32fb44c4da2))

* Merge branch &#39;ensure_return_type&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into ensure_return_type ([`7d09487`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7d09487f295a521d108fdea7f71f74a67e80bd13))


## v0.12.2 (2024-02-17)

### Ci

* ci: Added not planned as valid no stale label ([`a2dd834`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a2dd83476f24a2d2d064cf60225e40891c0a8a62))

### Fix

* fix: Removed translate-embed integration test ([`adb9cd6`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/adb9cd6a22c103cfffb8c703041e8dc00d919da4))

* fix: removing smaller translate then embed models ([`fbb9e97`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/fbb9e9731d22671829a2e7048050b31cbd3b034a))

* fix: removing smaller translate then embed models ([`91f6b79`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/91f6b791e0f084e29f8452469e43ebc23a8a8876))

* fix: Add missing scores ([`3b92090`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3b92090432800cfecc470748a626d92c503b5df3))

* fix: Added e5 mistral scores ([`7515e79`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7515e79710a4f518969d7b8b3f771f2592643138))

### Style

* style: ran linting ([`f729288`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f729288fefab8799eabac56efbb4896eb8c75bc9))

### Unknown

* Merge pull request #143 from KennethEnevoldsen/run-e5

Updated e5-mistral model ([`0026c9c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0026c9c79f643bdc6384c70152ce0258240f15d2))

* Merge branch &#39;run-e5&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into run-e5 ([`fdc19fb`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/fdc19fb0ddff979ca54bf1cf69d026ff94bb22cc))

* Merge branch &#39;run-e5&#39; of https://github.com/KennethEnevoldsen/Scandinavian-Embedding-Benchmark into run-e5 ([`e691448`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e6914480d3570b3901f4930bb59abd91a73bf0d3))


## v0.12.1 (2024-02-15)

### Documentation

* docs: Added dataset disclaimer (#145) ([`6b3e71b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6b3e71b29ab12b46b1e31eac654a493123346cc9))

### Fix

* fix: Updated tests ([`63d33c3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/63d33c33c6af4d585edc7f0a27e48b763429093a))

* fix: Applied linter and static type checks ([`b1baee9`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b1baee919a716431768160962305cf1d5f342c75))

* fix: Added get_documents to task interface ([`c4fb354`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c4fb3544710bc47781ccc6cd116eb54273b10e5c))

* fix: updated e5 model

fixed passing of batch size, ensure it can run on DanFEVER and avoid collecting to gradient (which lead to OOM errors) ([`189751d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/189751dce23a9fcb09b240478074f1df8e94a815))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into run-e5 ([`64e4986`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/64e4986e578aac977d56de185d2d8599fa18958b))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/Scandinavian-Embedding-Benchmark ([`24644a7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/24644a7f1febd022ce88c6cd0526b5162271f405))

* Fix: Added performance metrics for translate and embed ([`91a2b8a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/91a2b8addd67b66f9fd5d281c367e001842ee6f6))

* Added SwednClustering and Retrieval to cache ([`613ee8c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/613ee8cacda7efc58c081714b4dba3b63acea4f6))

* Added NorQuad and SNL_retr to cache ([`fe823c7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/fe823c79d24e247e015021b1f4a0b037dd689a61))

* Added a couple of tasks from e5 Mistral ([`b94648c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b94648c3906e37cc46f576abb9e4a2d889917808))

* Embeddings are sent back to the CPU, so they can be converted to numpy arrays ([`e154c39`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e154c393fe236068aba6df832f03316e5dc089b2))

* Lowered maximum batch size to 16 ([`f9b68af`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f9b68af2e0ca8992bc6598de8a34998edcbc1854))

* Added GPU inference to E5 Mistral ([`106084b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/106084b76fc230b23b43a5119ab50a99f4109741))

* Merge branch &#39;main&#39; into run-e5 ([`2fb8807`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2fb88070cf291951816a22a5e438e4f45b3f1e00))


## v0.12.0 (2024-02-12)

### Unknown

* Merge pull request #137 from KennethEnevoldsen/update-sonar-models

Update and rerun sonar models ([`28ff5e5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/28ff5e5bf8f68d6490b7dbc98cd1146f20eb974d))


## v0.11.4 (2024-02-12)

### Feature

* feat: Updated sonar models ([`f5f7374`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f5f7374ce31680aa6b8b3435035d9bd9e1c391c9))

### Fix

* fix: Applied linter ([`702f804`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/702f8049d27277ef86f5b91c87cc51d4a718887a))

* fix: Added model type and releae date to model meta

This is to allow the tracking of improvement on SEB over time ([`49c9f1a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/49c9f1aa9badaa702ae56361ef89cf50d22b0b38))

* fix: Added results of the sonar models ([`d0988b6`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d0988b6942626fcdb86e3d982aa9980601147a3d))

* fix: ensure that sonar model are proberly moved to device ([`746934a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/746934aafc7c79324a0ba89bcda3627961bab3f4))

* fix: updated sonar requirement to handle &gt;512 token sequences ([`792eb80`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/792eb802a2836c6ccb7d7f4d48307ad712ed7a53))

* fix: removed cache of sonar models, due to new update ([`c140b9f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c140b9f234262311574d0bdded699773b7d34cb3))

### Unknown

* Merge pull request #140 from KennethEnevoldsen/adding-architecture

fix: Added model type and releae date to model meta ([`245e881`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/245e88118d3972fe123c6fda361ffdf1f025160d))

* Merge pull request #131 from KennethEnevoldsen/mistral-instructions

Added instructions for all tasks in Mistral E5 ([`006c253`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/006c25308b3018e82be142f5bb884091529ccc21))

* Added instructions for all tasks in Mistral E5 ([`07c4c95`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/07c4c9572eff110605629dd6bbfab95df9c3786e))


## v0.11.3 (2024-02-06)

### Documentation

* docs: Updated table w. dataset descriptions ([`f955379`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f955379ee8c97ecd7916e3e9a767e3bc7dd555b9))

* docs: Added across column for coverage

added swapped formalization and task columns ([`31d9d76`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/31d9d76e257919c5541ccf1652ff4f48fe2bbe81))

### Fix

* fix: Updated dataset description metadata and script ([`0e63eb4`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0e63eb40a48bddafa0abff289c4972752dee1f26))

* fix: Update calc. descriptive stats ([`bdbd552`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/bdbd552ee28a118ce1a6da76fea8b6484828aeb4))

### Unknown

* Merge pull request #130 from KennethEnevoldsen/docs-update

Update task/dataset descriptions ([`144f025`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/144f025b7eb7d3b29af1e79662028fca758fd35d))


## v0.11.2 (2024-02-05)

### Fix

* fix: Updated table generation of the benchmark ([`c87aed4`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c87aed42be525a737426a820833ed3fd3d652fa5))

### Style

* style: ran linting ([`489e124`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/489e124ab368db33b0b69661a1e13cbb0dd3e7af))

### Unknown

* Merge pull request #127 from KennethEnevoldsen/run-models

Ran most of the models ([`db1868e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/db1868e9438ebd0f686e41a58bb0522efd0342d3))

* remo ve test srcipt ([`dcbe547`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/dcbe547fdd6fdbf9e9788fed49a2c2645eeed48e))

* Added translate and embed scores ([`26805ad`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/26805ad1052fced0402317281a362999848d927c))

* Merge branch &#39;run-models&#39; of https://github.com/KennethEnevoldsen/Scandinavian-Embedding-Benchmark into run-models ([`5570776`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/55707761d6f66fcdf218a38d9e50fc2f466591ec))

* Added fasttext and translate scores ([`d725b44`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d725b443fdd28d8f28500c58e6df6db4e8817f18))


## v0.11.1 (2024-02-04)

### Documentation

* docs: sort model on new tables ([`d4d9e56`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d4d9e564a207698279fd7575b583ad47286b31cf))

* docs: Minor grammatical fixes ([`633b47d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/633b47da12771f6da6234a428d329479c6817a64))

* docs: Added speed x performance tab to documentation ([`b3011a3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b3011a3e596e020068369dc6e504d575e4cbccca))

* docs: renamed run_benchmark to update_benchmark tables

Also disabled it actually running the model. Models are now run using `seb run` ([`da562b7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/da562b7f19298f9b33c921596480243fd9fb7a3e))

* docs: restructured dataset overview ([`30440f0`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/30440f07c853da518e936cb0b74dc0e39381788a))

### Fix

* fix: correctly check if model have a to() method ([`e2a1f44`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e2a1f44e9b361a7e277d6042f154ea6172121c42))

* fix: ensure that the correct sentence transformer wrapper in using the CLI ([`03fff7f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/03fff7fd63584df8703de0871bf5a01c2bdef7c6))

* fix: removed GPU test from speed test

at least for the moment ([`a7600bd`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a7600bd31490cd68805589f4bffd67e9267fd942))

* fix: Updated cache for models ([`ec07680`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ec07680e2c756bdf73f168bd618159e901ad27e4))

* fix: lower default batch size for e5 mistral model ([`3cdd55e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3cdd55e6261f8cebc75dd4098f329c9d8e64e59d))

* fix: added e5 mistral cache ([`ecc85c4`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ecc85c40d8938ddaf45782dd3979747c157176bb))

* fix: Added cache for all smaller models on all tasks ([`789c8b7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/789c8b798e5d36224201fe591df493436b9089ae))

* fix: added WPS to speed tasks and benchmark ([`7ebaca4`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7ebaca44815ab076cb70a29661b740702ceda1f2))

* fix: Updated scores for all API models ([`f10c01a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f10c01a045d4a689a3b7d504b1d762d67a5f70ea))

* fix: remove duplicate e5 mistral model ([`2de6ee2`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2de6ee23fc22dd2e21d035473adad2fa23c8accb))

* fix: Updated tables ([`d30284c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d30284c6bb62af1a8e14bc1db140278b994c0742))

### Style

* style: Applied linter ([`2b0342d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2b0342d6aeac41462f51360a6094279274b6ff30))

### Unknown

* tests: remove fasttext from integration test as it takes too long to download ([`84baefa`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/84baefa00c92356025cb6cecc8393b6c70ec1c78))

* ignore fasttext files ([`0f9823e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0f9823ec768675a46950e0057d2396e2081ee324))

* ignore fasttext models ([`4dbba98`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4dbba98138cffa4da4437e3fff76245ab82cdcd1))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into run-models ([`f13be1c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f13be1c2bd082a31ae09c7e6fee2f78f56110952))

* Merge pull request #121 from KennethEnevoldsen/update-tables

Updated docs ([`02b5993`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/02b5993b274c934fa948c0c26906bf3bc5a1c5cf))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into update-tables ([`db3ae97`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/db3ae9714ef799d544ccee876a4e65c71ca5aa9b))

* Removed req. for datawrapper api when running benchmkark ([`ee8673f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ee8673f4dba8591703e8a6258d6e5045ee87ff8f))

* Merge pull request #103 from KennethEnevoldsen/restructure_model_interface

Added LazyLoadEncoder, added SebModel and removed EmbeddingModel ([`4e1cd36`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4e1cd36e4a42079307393c59e905b99c30836989))

* Fixed linting ([`82c0a49`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/82c0a490a02a5a9f102b592a9273cf25e9a1644b))

* Changed Norwegian Bitext Mining revision to None ([`a5675eb`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a5675eb1ff74fc2c2a18d0c8742ebe1cddd26d67))

* Added docstring to LazyLoadEncoder, adjusted api.md to new interface. ([`c48e9cf`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c48e9cf2b6fbfeea6a81eceda6097240af0dfeb4))

* Changed model building to use SentenceTransformerWithTaskEncode in CLI ([`c85aad0`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c85aad0d08f2da8b9d7c88dc418f295d5e7c3327))

* Ran linting on danish tasks ([`3663cd8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3663cd839b9f8520de18ae91869fc301a244136d))

* Ran linting ([`8f1c3b3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8f1c3b36b432d1a2f0d958e28248b59a1cbd074e))

* Removed verbose parameter, as it does not exist ([`2a743d0`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2a743d093d38a66f1a4fcfb9caba6778396a5070))

* Converted new OpenAI models to the new interface ([`8b24890`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8b2489086ce920282b968e6b23ae968ad665f79a))

* Fixed syntax error in Cohere ([`e1d18f1`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e1d18f16054561b5eea92fd807a6d027c8003876))

* Fixed type errors in speed task ([`834e5c7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/834e5c7351d62a1e48119afc69531612418e3413))

* Fixed tuple.extend type errors ([`341930a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/341930acca86ea7f6ac16e4b180d82b2759e5f1b))

* Merge branch &#39;main&#39; into restructure_model_interface ([`7b20806`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7b208061dbcad56161b6155966b2326bdd30b76a))

* Merge branch &#39;run-models&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into run-models ([`9a80296`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/9a80296a53417208740635581484820ef22aba08))

* Merge branch &#39;run-models&#39; of https://github.com/KennethEnevoldsen/Scandinavian-Embedding-Benchmark into run-models ([`11734d0`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/11734d01ddec0df478c6e85e161965248deb11fd))

* remove accidental test file ([`25da0fe`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/25da0feac456319b789df28a1091452162cf64b5))

* added openAI scores ([`edff86b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/edff86bf7cee799eb44f2a6ae4743ed831ac7a50))

* Remove cohere scores ([`bcfdee5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/bcfdee56ff3a40b2d7932c40948f97a8cddab7c8))


## v0.11.0 (2024-01-29)

### Fix

* fix: Added cache for speed benchmark ([`f3cd21e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f3cd21e84db31e71b2accb49dcb43a7c89ce22e2))

### Unknown

* Merge pull request #120 from KennethEnevoldsen/ran-speed-bench

fix: Added cache for speed benchmark ([`b4596c3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b4596c3d6010a9e34c26beab210ad134ab4035ff))

* Merge pull request #109 from KennethEnevoldsen/add-sts-retrieval-dataset

Add SNL Clustering task ([`9d1dae4`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/9d1dae45d8959737a0a7e7126276d6b59c48f742))

* Merge pull request #110 from KennethEnevoldsen/sts-retrieval

Add SNL retrieval ([`9412d71`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/9412d71e164549f1bded845da815cd9cc0d8783b))

* Merge branch &#39;add-sts-retrieval-dataset&#39; into sts-retrieval ([`a1216bf`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a1216bfadd246a04517dac673de404e1f1403702))


## v0.10.0 (2024-01-29)

### Unknown

* Merge pull request #113 from KennethEnevoldsen/twitterhjerne

Added twitterhjerne ([`a32dea2`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a32dea264da3d3a71f8fb49167659e2fb47b0cbf))

* Merge pull request #117 from KennethEnevoldsen/nordjylland-retrieval

Added tv2nord retrieval dataset ([`1721983`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/1721983657cd004912c3e4ec78f25883256b26f1))

* Merge pull request #118 from KennethEnevoldsen/norquad

Added NorQuad ([`0cf610d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0cf610dc79c9a866f443fb3b4565e625ac868934))

* Merge pull request #119 from KennethEnevoldsen/updated-coverage-tables

Updated coverage tables ([`94f72d5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/94f72d5c072483f636e62a218d9a2321cb8614cb))

* Merge pull request #108 from KennethEnevoldsen/add-sts-retrieval

Restructured MTEB ([`115acef`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/115acef802c61cf057dc3fc0f07adaf55756e0d4))


## v0.9.4 (2024-01-28)

### Documentation

* docs: Updated coverage tables ([`3f848bd`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3f848bd02f84a3b705cffe0b3097eba67b78935b))

* docs: allow error for now in docs ([`4617983`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4617983837f222c06eec2a4c5e7b83ed094286d4))

### Feature

* feat: Added NorQuad ([`b1ab34d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b1ab34de84036dada7940742505b823aeff9d15b))

* feat: Added tv2nord retrieval dataset ([`c315d94`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c315d94d51795d3e62231997506e344731c6c98a))

* feat: Added twitterhjerne ([`8509c37`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8509c3733aeeab082e59dbad24c0e5b73fdab4f2))

* feat: Added SNL Retrieval ([`b1169ff`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b1169ff6a7f61d8a64f8246b5d38575e3bd0bf43))

* feat: Added SNL clustering ([`c380519`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c380519a844f56091975d6e3e57ae7ef44b66ef7))

* feat: removed swedn sts to experimental tasks ([`6d22966`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6d22966ada7bd8d5610b7562cd852d35813e0bf7))

### Fix

* fix: speed benchmark actually runs the speed tasks ([`7ae3ef0`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7ae3ef04e3d9b29c80e77a5f4ea57b4ca58cdb92))

* fix: Update wrong language tag. ([`6a143e7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6a143e725102fb3a5e99f8b55d622328fcd69d69))

* fix: remove ingress from the SNL corpus as they almost always contain the headline ([`617d616`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/617d6167479d2955f6034bd1fc6b3e4810eb591e))

* fix: restructure mteb tasks  to its own folder ([`d88864d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d88864d68f7243dfd36921ab8bbd25f3a1efdb90))


## v0.9.3 (2024-01-28)

### Fix

* fix: Added new OpenAI Models ([`e096ef5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e096ef5efe351087207e49a254e3ffc6dcf30916))


## v0.9.2 (2024-01-26)

### Ci

* ci: Updated lint workflow to actually fail when not linted ([`d1c177c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d1c177cfe8c4f976ab7d8d8701b0df9fa80496c9))

### Fix

* fix: Added relevant type ignores ([`27b8dd3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/27b8dd3d52867e03f528390f361fc3b8b8f116f4))

### Unknown

* Merge pull request #104 from KennethEnevoldsen/ci-lint

Ci lint ([`ade460b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ade460b3c6beed2aff8589919aef19551980841f))

* fixing linting error ([`1b75a4f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/1b75a4f65d1f927355341ae61ba7731fd521c21c))

* introduce intentional lint error ([`d861935`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d861935b62b457b1b49c1dba74d6842448b8fdce))


## v0.9.1 (2024-01-26)

### Fix

* fix: ran swednsts and reduced dataset size ([`6c0a030`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6c0a030a7244914548f0bbc12426505cb7427555))

* fix: ensure that metrics is correctly formatted from MTEB ([`b5873b8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b5873b8ef8471afb82d8f26c7a1dee5202d84f78))

### Unknown

* Merge pull request #100 from KennethEnevoldsen/sts_vs_retrieval

Reduce size of SwednSTS ([`c2c32b7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c2c32b7504fabc5b43d198edf5ec6f5546acf310))

* Added LazyLoadEncoder, added SebModel and removed EmbeddingModel ([`805c343`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/805c3432099d3426fd8ef15c87bd54522a763b30))

* fixed type hints ([`0a5fd27`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0a5fd275439effcef106d7eea9b4e2aa9270b300))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into sts_vs_retrieval ([`141ed73`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/141ed73f6ad066f94273596c497a2e5b3bc76022))

* Merge pull request #89 from KennethEnevoldsen/stuff_runs_tests

Added integration test for four model types ([`f427a6d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f427a6da949fb2f7eb14f1014e9ae7a43df3a109))

* Merge pull request #92 from KennethEnevoldsen/custom_embeddings

Custom embeddings for E5 and Cohere + Interface changes to accomodate this ([`aeb32ce`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/aeb32ce9b8105e7e5b71fa10297bb799246225ab))

* Reset test cases incorrectly overwritten by a merge conflict resolution ([`ccdd886`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ccdd8863fce445f39e21550806b4ca875a18215f))

* Merge branch &#39;custom_embeddings&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into custom_embeddings ([`2414337`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/24143375997a8c99e84d11d96177ee1308798e8c))

* Put resetting the model&#39;s encode() method to a finally clause ([`34a2612`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/34a2612eb578c909d14266092a539537e208eea9))

* Merge branch &#39;main&#39; into custom_embeddings ([`18cd858`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/18cd858c1349f44d8b404eacf52c8c983a4065e5))

* Removed debugging print statements from E5 ([`cc12070`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/cc12070004fb4ea943a5e7f76a086cba234ed8cc))

* Made EmbeddingModel into a dataclass instead of BaseModel ([`75debec`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/75debecb9e4108b30080845ff81bcd4935e81286))

* Removed reference to MTEBTask from ScaLA ([`674a005`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/674a0054e281d5536ac2df13b39e8f3ecd017c7b))

* Replaced MTEBTaskModel with partial() ([`79708bf`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/79708bfe5824a465ae018c4d0b4921acc9b47ff7))

* Added encode_queries and encode_documents to EmbeddingModel, made task optional ([`ecee037`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ecee037d73448de50320d58063d08917fcd9d77b))

* Merge branch &#39;main&#39; into stuff_runs_tests ([`01e8979`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/01e897997279bf092a54b993cc497a0ea25e4fc8))

* Moved models to @parametrize ([`fafcb39`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/fafcb39b3eae7e3921060b38f986da90531d20ab))


## v0.9.0 (2024-01-26)

### Feature

* feat: Added performance metrics for danfever ([`22eb72b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/22eb72bc1b530d3425d2b4fd1fb64d3bf3b2c971))

### Fix

* fix: limit the size of STS ([`0d1b659`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0d1b6590d69bddbfda5154827b6a9341ccc8cec5))

### Unknown

* Merge pull request #97 from KennethEnevoldsen/add-danfever

Add danFEVER ([`801753f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/801753f0d5500e91d935093140daaa402b8d4b57))

* appease pyright ([`a572962`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a572962c5f6fc6826182c86aa5b6ad5ec1af076e))

* tests: remove tests which has to be changed when adding new datasets ([`04aa44e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/04aa44e5c62e26919eddd16487f92c37b85f6fb4))

* tests: convert test_task back to normal ([`be2c071`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/be2c071f85fb2454bb4b90e1dbad8602eef6d4bf))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into add-danfever ([`69a5a03`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/69a5a0367f5025a1d8184ceee0589549d7872a32))


## v0.8.0 (2024-01-25)

### Ci

* ci: fix mispecified yaml syntax ([`ca5567c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ca5567c71d3da1ec49b59c0a74b74bd85219a46e))

### Documentation

* docs: formatting code blocks ([`cee41f3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/cee41f39ad3b8a9c4095399c28abcc2b0704f229))

* docs: update docs to not run all models ([`90cef3d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/90cef3d05053f4d007c3ed926240243a9c5ecf93))

### Feature

* feat: Added danfever ([`ccec57c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ccec57cf67f575f97751b7d87d36a273825b9606))

* feat: Added VG clustering dataset ([`49e75d5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/49e75d554e43b48f99b6257b2ae0bd2b4cf828dc))

* feat: Add swedn clustering ([`0786ec5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0786ec5a96e7607cbaf7efd368724b1fb9b6d043))

### Fix

* fix: Update indexes to strings ([`37d165f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/37d165f162a324c33c180d767bcdbdc9765e972a))

* fix: fixed error arised from merge ([`11e28d6`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/11e28d62b27b0654bf683c525f4dab2848995160))

* fix: updated based on static type checks ([`4752f07`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4752f070ac711aba0514de8f33d8511e3da8a274))

* fix: move description to the end as to make printing of task object prettier ([`f8ec70d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f8ec70d2125cac3b01c5717c8d7d0f73159acb28))

* fix: reduced size of SwednClustering and ensure that clusters match with document size ([`0b70730`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0b7073082da8340b6cad68fb91121b22329355b0))

### Style

* style: ran linter ([`bcc1231`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/bcc123179a2ac0764f00d223b900b4840fa90da5))

* style: ran linting ([`05b6bf9`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/05b6bf9828cb3327f6b5ab231f6e18e5ebdb5c6c))

### Test

* test: Performance using 5x2048 examples is 8.13 ([`ed5cb5d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ed5cb5d9bcbda66a2eb9797e1c280a7af0243d80))

* test: Performance using 5x10000 examples is 13.80 ([`ed36b82`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ed36b821e3e65e48468d48163936318dea401cd9))

* test: Performance using 2x10000 examples is 8.70 ([`6fe30b7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6fe30b7e002c4ce7bfbe7f48611e4f17821f77e5))

* test: Performance using 10000 examples is 8.46 ([`630769c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/630769cb8acafb972bf82254217eba83b96f6773))

* test: Performance using 1000 examples is 8.12 ([`7732c32`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7732c3279533bc389fed434085c5ab699d53f0a4))

* test: Performance using 100 examples is 21.07 ([`82f7b3f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/82f7b3fd67a4f866f5899f49b5ceca7df0333c64))

### Unknown

* Merge pull request #96 from KennethEnevoldsen/add-swedn-clustering

Add Swedn and VG clustering datasets ([`8537e12`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8537e121d115522cec4eaaef8b2d501606d7cdaa))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into add-danfever ([`796e3c9`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/796e3c993d789ad5ddc7ea050fc3f08dfd80b249))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into add-swedn-clustering ([`18f9afb`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/18f9afb722e353b7f2ed6cf3452935c0cbf70cfb))

* tests: refactored tests to not be highly dependent on a few tasks ([`4b1eaa5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4b1eaa556db1887a995db52e0f49da1edca42217))

* Added a bunch of experiments for the vg summerization. ([`d9a13cb`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d9a13cbc0d5932ebc1153da03e15a4db73ed3451))

* Added task-dependent and asymmetrical embeddings for Cohere ([`6742fdd`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6742fdd4b710fd5b1e7ca3145454f60b4330574f))

* Added encode_queries() and encode_corpus() to E5Wrapper ([`53390da`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/53390da85b1b9462504568f9dd0e769a6c485407))

* Added integration test for MiniLM + LCC ([`4560a16`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4560a162ec85f66ba961afad9afba7e43a2a274b))

* Added return type to appease linter ([`828e556`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/828e556d1626e759cbf711ef8d06ad98d376689e))

* Switched out fasttext package with fasttext-wheel ([`2cd74d4`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2cd74d4d9e167d2b8380dfd3947689a9a211fc25))

* Added pybind as an optional dependency for fasttext ([`77cdd4b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/77cdd4bde812d527a63f67f2b874f28f4844af7b))

* Added fasttext as dependency in the makefile ([`012a689`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/012a6892d9a01a1c09d6308b0d21208c605e7ae3))

* Added Dummy task to integration test, remove all-MiniLM-L6 ([`2f75712`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2f757121b8ef77fdf0120a6df3d3394dc19b5d1b))

* Fixed issue with fasttext models ([`0a015ba`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0a015bafc96439ce04df55f90707f0fdff9819f6))

* Merge pull request #90 from KennethEnevoldsen/types

Moved task types to task interface and deleted types module ([`7c3b582`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7c3b582943b26060303816f70d122930083ac70e))

* Added English to Language type ([`221bdd8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/221bdd80bde334ec077e75f0f014b9b48e75de03))

* Added pybind install to makefile ([`31d2a35`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/31d2a354a6592da870492b47dc589b7c92b140e4))

* Removed faulty import in E5 models ([`601002c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/601002cc9613a59cb9bf236fab909d2bd24433d3))

* Added fasttext to testing dependencies ([`045de9f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/045de9fc453fb3136d79b374c5e7337a0ebe20b2))

* Renamed test function ([`3c50f4f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3c50f4f1b11dc6d873964171655ac92fd6b76ab9))

* Moved integration test to new file ([`6d3ba6d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6d3ba6de3ba929471a4fb3d0586491fcdde09877))

* Merge pull request #91 from KennethEnevoldsen/new_models

Added Jina base ([`95c515e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/95c515e7c6095f75f825692f141d622dea9ff3e1))

* Fixed import error in speed task ([`cfccbdf`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/cfccbdf3aa32e8b73e4d89ac924ac1eefac64986))

* Added Jina base ([`6d1ec69`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6d1ec694f9b53e76f86e97ed4df4be70717f465b))

* Changed DKHate to LCC ([`8c004af`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8c004af42005a6cf826d962ff40678fe96789d8a))

* Moved task types to task interface and deleted types module ([`2f1adf1`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2f1adf1305567b7f989b288344cb32caef7efad5))


## v0.7.1 (2024-01-23)

### Fix

* fix: added task argument to TranslateE5 encoding ([`71dcd09`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/71dcd0994df98e891a80c1576b8bf0d8411a151d))

### Unknown

* Merge pull request #87 from KennethEnevoldsen/new_models

Added XLM-Roberta large and LaBSE ([`74fcf43`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/74fcf431d2f7824ecb570722f4fccc0262f41615))

* Merge pull request #85 from x-tabdeveloping/main

Added FastText and Translate-E5 models ([`2d9043e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2d9043ea2f52243de168f110bc9732dae4abf14c))

* Removed commented-out lines ([`373b937`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/373b9378f7e4e75c73e527c5741932f7faca8ce9))

* Removed duplicate model ([`476d679`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/476d679e10aaf1b5c0d123d2cca1d9ed12440bdd))

* Fixed duplicate model names ([`7275686`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7275686b4a22972b803b76dffa92aa9f9cfbe37d))

* Added XLM-Roberta large and LaBSE ([`858db1b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/858db1bd3e305c3d9299ffd892aa14841608985a))

* Added integration test for four model types ([`61ff3bb`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/61ff3bb13af50620ab55f1f6004ef91e2f14d45e))

* Merging upstream into the branch so that it contains the fixed E5 models, that pass along the task. ([`f6f71db`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f6f71db4648867e0118af35aa75b65018bd5cc77))

* TranslateE5 now uses E5Wrapper to ensure task-correct embeddings and prefixes. ([`5eff1fc`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/5eff1fcad96d15fdad962b370568ae9c676efa8d))

* Translate now returns a single string instead of a list ([`e1cefd9`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e1cefd919fa69b9c9edf36eb960e258613322d2b))


## v0.7.0 (2024-01-23)

### Feature

* feat: Added SwednRetrieval task

The idea is that it can be compared with SwednSTS to which one makes the most sense. ([`7fe3371`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7fe337195a98a6a97c0e590401fd47ce2158334c))

### Unknown

* Merge pull request #82 from KennethEnevoldsen/add-retrieval-swedn

feat: Added SwednRetrieval task ([`d5f959d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d5f959d7fca26bc1bc5c09769d7aca0dcb0b7e28))

* No longer imports ..types because of an error ([`c75629a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c75629acd9a2a74c404f0a0cde2e5ea26df7f40b))

* Fasttext now loads on initialization instead of the first encode() call ([`e6b1fda`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e6b1fdaf611ed413d7b5bcd649c694cb77196801))

* Added Translate E5 models ([`33835af`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/33835afa0a5fcc4c25c54dc4424a3fa2bf63358d))

* Added fasttext models to cache ([`fbc9482`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/fbc94829648ec40be94a8ca5bb65b84f0ab78a2e))

* Fixed model names ([`d28f259`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d28f259e428e0b85829fbde9a4cf3e6c16b370cc))

* Changed model fasttext names ([`7871ab5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7871ab592f4911ab6197b6d62be8a7e2c897e060))


## v0.6.0 (2024-01-22)

### Fix

* fix: Allow models to batch inputs ([`09c3527`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/09c35278018671ed21e71f8ee749b2b12127efab))

### Unknown

* Merge pull request #70 from KennethEnevoldsen/add-speed-task

Added speed task ([`d192e44`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d192e445956c979e6e2e806cf76778555c0e09e4))


## v0.5.5 (2024-01-22)

### Fix

* fix: Add toggle for verbosity on the cli and remove duplicate entries in table ([`4d26fce`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4d26fce32019b61dad08fc5adeb9c47ca85b9226))

### Unknown

* Merge pull request #74 from KennethEnevoldsen/verbosity_for_cli

Fix verbosity toggle on CLI and remove duplicate entries in table ([`99ef0f2`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/99ef0f2886275158aa43952f95d7f9aa1c9a5a16))

* Fixed newline error with FastText ([`65411a3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/65411a3c50096e0099925ee66f00b76a7604f3ea))

* Remove model results for repo ([`2435011`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2435011311ef423f27d72f3706675cbada8ade38))

* Made fasttext models compatible with the new interface ([`ac8d46f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ac8d46fa0329112cca8b966d3904f08a69a69804))

* Merge branch &#39;main&#39; of https://github.com/x-tabdeveloping/scandinavian-embedding-benchmark into main ([`c0be7a6`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c0be7a650b21dae93966c9f49784444e32a2701a))

* Added fasttext models for nn, nb, da, sv ([`1820689`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/1820689e3af0f0d6673c18290dd8e7a7045642fe))

* Added Fasttext models for nb, nn, sv and da ([`184dde8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/184dde8c7c0d17fdb7bcb2bbeab83c76000eb94a))


## v0.5.4 (2024-01-22)

### Fix

* fix: ScaLA now correctly wraps models to allow for task argument to be passed ([`3b07a4d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3b07a4d9cf8ad1170453feb3f74fb065edd05114))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`07efe8f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/07efe8f29fc14d3db50e96a91283b3e16dea77fa))


## v0.5.3 (2024-01-22)

### Feature

* feat: Added speed task for estimating the speed of the embedding models ([`25caacc`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/25caacc87baed6f3c85d6c40682a28790409dc04))

### Fix

* fix: ScaLA now correctly wraps models to allow for task argument to be passed Renamed scala cache ([`a70c950`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a70c950b0093924e9aa64dc4c0a4604cb868c864))

* fix: fixed a type hint in tests ([`da32c0e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/da32c0eb09982be8b6b2a40c2a6d195f11f6d428))

### Unknown

* Merge pull request #73 from KennethEnevoldsen/bug-scala-missing-task-encode-wrapper

Wraps ScaLA models in MTEBTaskModel ([`e2eee05`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e2eee055775ffcc7019fa470dcd5d70a124416aa))

* Merging with current status of Upstream ([`9c1fdf3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/9c1fdf3cbea2f63eec45e051d7330678d6da11df))

* Added Fasttext models for nb, nn, sv and da ([`4249333`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4249333e9b2b89c1c8900fb80c2f677723fa6279))


## v0.5.2 (2024-01-19)

### Documentation

* docs: Added norwegian courts to table ([`4f31602`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4f31602b22d4327c7853f05fc04a03a04dd9ff1b))

### Unknown

* Merge pull request #66 from KennethEnevoldsen/allow-custom-embeddings

Allow custom embeddings ([`698453a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/698453a9eda4cb9661b6751dae1361341b97e7f8))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into allow-custom-embeddings ([`f8b91f1`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f8b91f161b2fe73b8aa539c85fb15e10f678cc3f))


## v0.5.1 (2024-01-19)

### Ci

* ci: rename pre-commit to lint ([`1f3c743`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/1f3c74368341ce08e398aadd6c4391bc5628352a))

### Documentation

* docs: Updated CLI docs for run ([`8ff59b3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8ff59b391e50d7a631d95ebf63a5fff4a834b83e))

* docs: make sure that tutorials are tested on prs ([`e4ef73a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e4ef73a1d94f9dfbecf6c739adb4c12e5d63e4dc))

* docs: Updated tutorial with the CLI ([`525eaf2`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/525eaf27f46f1ff036d5bb237e7e645b96270ce4))

### Fix

* fix: Run command now print table for target models ([`b7d444b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b7d444befad022fed23f92a30d6dec94413348cd))

* fix: Benchmark result now save in the same format as the cache ([`896c3bf`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/896c3bf553dfe3e7738b47f2a70525a661ab0940))

* fix: updated according to static type checks ([`f3c77aa`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f3c77aa873d788b8b61271daa614833da492c680))

* fix: Added missing init file to make sure that docs build ([`8d44640`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8d446409e1b58f1a432b676e1dd6b0fad521c2cc))

* fix: require positional argument for encoder ([`a7040a5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a7040a5385124945d4697a519b47182c9d668289))

* fix: restructure repo ([`64bace6`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/64bace65f840ffb8c11cc9ba689b4509d4e18e26))

### Unknown

* Merge pull request #65 from x-tabdeveloping/main

Added Norwegian bitext mining task ([`2e8bb07`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2e8bb070fb4b4128c3491dc6995ff171f74c9cf3))

* Merge pull request #68 from KennethEnevoldsen/cli_updates

Added table to CLI ([`2457782`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2457782374aa402b10623626c22570610b9e6510))

* Merge pull request #69 from KennethEnevoldsen/KennethEnevoldsen-patch-1

Update paper.md ([`1676c66`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/1676c66d25e33e4195418314e4ec0cd3bf0d6a10))

* Update paper.md ([`eca5a43`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/eca5a4377a8b9966148a46941398bb5cf39b7220))

* Changed ruff formatter to use line length 150 ([`1079489`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/107948928fda0476c89b8bd816164fd0c6f240ce))

* Merge remote-tracking branch &#39;upstream/main&#39; into main ([`d0739d8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d0739d8b3d03c9dd2cc6470141bbac9c91fc9a44))

* Merge branch &#39;cli&#39; of https://github.com/x-tabdeveloping/scandinavian-embedding-benchmark into cli_updates ([`7ef411f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7ef411f81dcc6e4643493b4d35b989dd3a0c663f))

* tests: Tests pass as inteded ([`d19c650`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d19c6505e4bf7cee055b9568cc7dd8deab61525b))

* Altered tests for CLI ([`e4f5457`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e4f5457be47b46a4ca1a3234f782db6a4e3dcc51))

* Tried fixing type errors (ignore problems that are not actually problems) ([`6123138`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6123138b89742a12872b4638d75409985e134c97))

* Changed -h docs to reflect changes in behaviour ([`b8b8169`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b8b8169528e2c1c1f59539b32fc810d6493bd1ba))

* Model printing nicer with less space, fixed multiple arguments, and implemented new output interface ([`fdd072b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/fdd072b0def344fa0a14457003928e2b493b9d64))

* Made model and output path optional ([`fb1c976`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/fb1c976408c2acc444b7f48ab7e083c66123bfff))

* Added more direct reference and commented out task ([`566a009`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/566a009d221501ddc19c672343460189ebe59d15))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`af6f926`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/af6f926163bca4d68689d44e3bedd169bc6c0d7a))

* Added Norwegian bitext mining task ([`49b1655`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/49b1655b331fbc9f733dd883a8b27a19926a1d00))

* Moved more code into main CLI, pretty printing now takes DataFrame ([`ae1cb90`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ae1cb9005c7162eeade95d861673096917e7b9f7))

* Added pretty printing benchmark results in table to CLI ([`eba47c1`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/eba47c1233ab972ff816a72583e3b698c10d159e))

* Merge pull request #62 from KennethEnevoldsen/update-tutorial-with-cli

Make sure that tutorials are tested on prs ([`ebf1640`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ebf1640a05511fd6a145dba51596b2d20e9345e2))


## v0.5.0 (2024-01-15)

### Fix

* fix: Fixed errors derived from merge conflicts ([`5f086cc`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/5f086ccff5b588888ca9bc47471ee706e1239101))

### Unknown

* Merge pull request #60 from KennethEnevoldsen/add-summarization

Add summarization ([`065b0e6`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/065b0e632b188362085705f9809409b055683dc7))


## v0.4.0 (2024-01-15)

### Feature

* feat: Allow tasks to be passed to benchmark instead of just strings ([`57a9b19`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/57a9b194a741d0416a9289fe11da174835cfdd63))

### Fix

* fix: remove commented out code ([`04187b3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/04187b35f252c64e7ddb0a31e75ed6df03902206))

* fix: remove DKHate from tests ([`3192fff`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3192fff27519a48aadcb46e8f126c4894e3ec727))

* fix: Renamed Scala -&gt; ScaLA to ensure cache hit on non osx system ([`d9f7b05`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d9f7b0575dd140d840e9104755408c49c3e22b2e))

* fix: Added intfloat/e5-mistral-7b results to cache ([`93b0086`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/93b008617a9717c8011cb34a3c1949d1d3e2c7df))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into add-summarization ([`5072584`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/5072584d261f0d6b4594a7487d2c75e77ac30312))

* Merge pull request #57 from KennethEnevoldsen/move_cli_to_radicli

fix: Added new and more comprehensive CLI ([`2b0a47a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2b0a47aa8ff3e044d15725adf6b20a2bb883aa69))

* tests: Fix ordering of test input ([`971c9ac`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/971c9ac7d0a4640e982fe62ebfd62a0372b8f22e))

* removed files ([`1e78ada`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/1e78ada48460b33592fcaddad83745a6ce260ec0))

* tests: Ensure that dummy tasks are not added to registry ([`affabc8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/affabc8a8aafcb16293079d464ddbd3867ab0f59))

* Fixed type annotation to 3.9 ([`cbb73d6`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/cbb73d6669d2f127f2a6e874858b1bfa9220f131))

* Merge branch &#39;move_cli_to_radicli&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into move_cli_to_radicli ([`76143d9`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/76143d98cb4d39356996e3f2413c35e1a7054ba2))

* Updated cookiecutter reference from Martins to Kenneths swift template

This old one is no longer udated. ([`034b428`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/034b428b5fdd7386ed97bb206aa51213b94093df))

* Updated cookiecutter reference ([`71b7ead`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/71b7eadaf5364ba3eca3cec0ad46bd5ae4c15eb2))


## v0.3.1 (2024-01-15)

### Documentation

* docs: added execute flag ([`ea0e9ca`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ea0e9ca6329a062c229d6157005f68cca4b98f15))

### Fix

* fix: Added new and more comprehensive CLI

Including documentation and tests ([`14ca469`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/14ca469241d6a6b356d567fd888f69b613e5652d))

* fix: SebModel -&gt; EmbeddingModel ([`d2f9efa`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d2f9efa1e0a367f6240aef7f0cad5ba4a9b56b11))

* fix: Allow embedding size to be None when using CLI ([`c621a8b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c621a8b328b1f6063cbd29d8a43376a9f587a00e))

### Unknown

* Merge pull request #58 from KennethEnevoldsen/update-cruft

Update cruft ([`870e442`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/870e4424f4448c948a2f14f1d9035276b63e04a0))

* clean: removed pre-commit as it is no longer used ([`f1b5804`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f1b5804b83796694157f282aacb1280986c58d01))

* updated from cruft template ([`bd7b11a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/bd7b11a202cca340103163b26e809e9491fea52b))


## v0.3.0 (2024-01-14)

### Ci

* ci: Updated some names in the workflow ([`b7c3012`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b7c3012d843d5eaa2017b96479f6ac730cd48087))

### Documentation

* docs: Added &#34;government&#34; domain to LCC ([`232dfee`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/232dfee8517890cec21e075dd4b79f524c928d86))

* docs: Added documentation of dataset coverage to datasets ([`e4f9468`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e4f9468140038c6dedf29f82a8b0b8dbf9c702c2))

* docs: Added avg rank to benchmark table ([`15a821e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/15a821e6ade440fc2a2bba218e0c91d8bbc3c480))

### Feature

* feat: Added e5 mistral model ([`56c0971`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/56c0971341034ae85a8499ce94e0eb0343355efc))

* feat: Added option to not run a model ([`55cd023`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/55cd023cd69be59e7444bf19c331ee89b4c5003f))

### Fix

* fix: Added Swedn dataset ([`ac6e744`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ac6e7442f15fe935ecf8718c8212acdccf804939))

* fix: refactor utils script out into its subcomponents ([`f36be90`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f36be9020dbb3fcfcfec3fbd302a61e274627122))

* fix: Allow optional embedding size in ModelMeta

This makes the possible to create a on the fly models using the CLI ([`b0b4793`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b0b47930d74a6c012fbdf6b85d04c1d6665aa888))

* fix: Updated CLI to now use models which is a part of the benchmark before wrapping it in sentence transformers ([`2b60c28`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2b60c288edd115a8944d21b66e657ae17dac5943))

* fix: Added embedding size of models ([`2937099`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/29370992610b78055f27a8c3ad7504147acbff5b))

* fix: Added mistral current scores ([`748d8a9`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/748d8a97d7a9f1d5f186cd2d92468167f912e6ac))

* fix: Added prettier prints when running benchmark ([`012bcd9`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/012bcd9def66cfbfe203e6e7fd346cd09f7e2594))

* fix: Added option to ignore cache ([`8f36080`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8f360805038ade5b6ba91a9726918b20ac0be6f6))

* fix: removed typer dependency ([`e519917`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e519917331218e993b9cc040851cb8d5d0097fdc))

* fix: removed duplicate on update bnehcmark ([`ef6270c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ef6270c19ff3086da8e6596f7ffc03fdce6a0038))

* fix: Added cache dir to all entry points ([`3fb4280`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3fb42806ffd9d0ccdbdcee45171d20a1390a8e68))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into add-summarization ([`8ed0a0d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8ed0a0dde5dc0bd35901927577fa8960fd64080a))

* Merge pull request #52 from KennethEnevoldsen/add-dataset

Add embedding size to benchmark ([`d40a633`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d40a63385db585121cea606b1c0505099855cafa))

* Merge pull request #50 from KennethEnevoldsen/run-using-cache

Add public cache to benchmark ([`67e571c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/67e571c3d0133dde1192c56ee21cb93afa4e2f3e))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/Scandinavian-Embedding-Benchmark ([`6d9d7c1`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6d9d7c180208f62532a69e010ac9a6235af20385))

* ignore files ([`a1904ed`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a1904ed2afeb12c2ecb71a66a2b9946303abbaa5))

* Added make command for table in docs ([`0cb3522`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0cb352299cd0ffba6c80edf68158d0157cca58fe))

* Merge pull request #51 from KennethEnevoldsen/run_mistral_on_ucloud

Added command for running on ucloud ([`396f79b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/396f79b79c9461ffb28b2b98ac1d95218a0b65af))

* clean: remove test file from cache ([`89bb78f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/89bb78faa9902822c7d3c4952d1d0e77aa0ec2ca))

* clean: removed test models from cache ([`10413e1`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/10413e1dda23bcb7741b6e96edcf871086f1e4e0))

* clean: remove tests from cache ([`af1f52d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/af1f52d540fc6b951e3a2453502b00853420e33b))

* Added test for checking if benchmark is up to date ([`8fa2545`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8fa254579d78e0dd4b1e7a9e857bc9ee87e458d0))

* Moved cache dir to package ([`94d6468`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/94d6468daf6ce2b33a53aa1485818bf38fafcaf2))

* Added command for running on ucloud ([`c48b32e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c48b32ed0b3863c10db06cd96f05bf098eb2a47a))

* Merge pull request #45 from KennethEnevoldsen/updated_norwegian_parl

Updated desc. for norwegian parl. ([`985dd5d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/985dd5db69819a7c865d5094bd63df8e908a7c3d))

* Updated desc. for norwegian parl. ([`c7f1e74`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c7f1e746dfa5596d63666fa994b1a951f3aaa573))

* Merge pull request #43 from KennethEnevoldsen/add-mistral

Added mistral dependencies ([`a4decba`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a4decbae1d6c9f3bf84f0e665449d514cd365779))

* Added mistral dependencies ([`1d39008`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/1d39008c744139b788fd43b6426fc517e0355d12))

* Merge pull request #40 from x-tabdeveloping/main

Added E5 Mistral ([`c317ad8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c317ad824e3016bac8cb860f2e8feedfe71bfc65))

* Added E5 Mistral ([`01cfb90`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/01cfb908e592efa5068ff9715dc18e14c767d05f))

* Added open-source flag to danskbertz ([`8ef4e39`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8ef4e39adf9e022ec046a80896f9510c0ec43390))

* Update index.md ([`41e350e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/41e350ef7c3951a3f902188c30e91ecd35ed78eb))


## v0.2.10 (2023-12-07)

### Ci

* ci: ensure full install in test ci ([`6144791`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/614479197ffba5951d49501e5f0be8902fec7adb))

* ci: updated docs ci ([`4e3b89f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4e3b89f5d0966dcfa7ada288a4c6246f0a30b825))

* ci: fix python version ([`dec0de6`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/dec0de69eda4d0805b591084d6e36a4820d7130b))

* ci: remove cache from docs ([`b53fa67`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b53fa67c9dc4334ac1a7610cd2d24af9ee43d77e))

* ci: lock python version ([`f00d8e8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f00d8e8877b06cf28c035ada2455c27f1af35b69))

* ci: Update pip and invalidate cache for doc ci ([`7c507a4`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7c507a41b3977d17d31eee191389bd6714cbe872))

### Documentation

* docs: fixed size of tables ([`f889199`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f889199df5b7e01220036ae8054ed5c38b9a5b28))

### Fix

* fix: ran ruff formatter ([`1d2341c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/1d2341c5b9bf4df95bd6513acb21516a3d5f45fc))

* fix: change to relative import for tests ([`3ec0673`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3ec0673f33a1270bff0b4ba8ccdcfb79784d838b))

* fix: Update from cruft template ([`5e055da`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/5e055daa2c7c9c8dfbd80aedebb881f24511643d))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`07c5edc`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/07c5edc87338da24982d98ed58242e72d489bd35))

* remove invoke from repo ([`90d7b0b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/90d7b0b68bdc5d4dbf19945691e86b7d4a52ba37))

* Merge pull request #39 from KennethEnevoldsen/update-cruft

Update cruft and fix cruft template ([`c2bf0cf`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c2bf0cf515e838f236627b1c343c7965aa9d62b0))

* ignore type error ([`eb0b647`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/eb0b6476e3f1b5c7c02e2567a6d9e62d492469e7))

* add missing updated to the makefile ([`ec857ab`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ec857ab924ebf1a4efff7b58630607cd013810e1))

* Update cruft ([`8016bd5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8016bd5f31f760ca5def957fffe822706b29de9e))

* Merge pull request #38 from HLasse/patch-3

Update citation.cff ([`0b099be`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0b099be3de7d85b5819ef3190c8bada05fbfadc6))

* Update citation.cff ([`b7fad4e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b7fad4e6a146dd05e370b33ee4d02c24270766f7))

* Merge pull request #36 from KennethEnevoldsen/KennethEnevoldsen-patch-1

Update README.md ([`8f3f9f9`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8f3f9f975d931e50f001b7c13e5e067b668a09e9))

* added citation cff ([`e76f244`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e76f2440e5b7d9c702c1f726f5a76e76d300f147))

* Update README.md ([`0730366`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0730366b6a6667bbceb72a5e53b96bd1a584fc4f))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`9176011`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/9176011968e633d65ce0c6f6d3cdecd1fd0ad091))


## v0.2.9 (2023-11-18)

### Build

* build: Updated dependencies for mteb ([`f300ac5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f300ac53146edd3201388f6948b5fd25ff1b705d))

### Ci

* ci: Updated makefile ([`892d72d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/892d72d6ac727706da82e1dc5211be2d9eaaa94a))

### Fix

* fix: added type ignore to optional imports ([`febfffb`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/febfffb11edfe736400d37db4f5e74fa2bd5a5e9))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`7d792e7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7d792e74ae132c27b375a8bbfe0ac24bcd47b709))

* benhcmark: Fixed ordering of columns ([`ba64303`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ba643036933a08fb0001512a6f8d3ea8e108de13))

* benchmark: Added open-source column ([`621377b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/621377b33acc3c4d95e7f247ef4964b93f614c0b))

* benchmark: Added globe for multilingual models ([`1a53419`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/1a53419792b02c67fd3f7702f1dcbb5101d22cfe))


## v0.2.8 (2023-11-17)

### Documentation

* docs: fix missing line ([`06dd1ed`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/06dd1ed489551f9176ae8466842c9bdd8dc2b0b5))

### Fix

* fix: ran pyrigt ([`e470a71`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e470a719ab39f59eb4000b9cb5a3b95ae6380c09))

* fix: remove openai and cohere out of main dependencies ([`b023a88`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b023a882f4b6cf4ba15bbbb07dd413ee49936344))

### Unknown

* Add cohere library and update task domains ([`ca4f8b8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ca4f8b8b58135aeca8eeabd42d6326d14570722a))

* adding desc stats script ([`197c0d5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/197c0d584ac23cc8bd5f307e97a089b9b9e423f6))


## v0.2.7 (2023-10-24)

### Fix

* fix: ran precommit ([`cf42195`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/cf4219502c32720204ae1ba79cf7d664066f8821))

### Unknown

* Merge pull request #27 from timpal0l/patch-1

Update hf_models.py ([`a72ac2f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a72ac2f4d19888fc46c4f0094a49381407d7da42))

* Update hf_models.py

Added `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` ([`c4d5886`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c4d5886642b71f3fb965ba0ecbcf796345d4b229))

* Merge pull request #25 from x-tabdeveloping/main

Language selection in CLI ([`4cb3f7e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4cb3f7e4ed0f745b46a0c8a8af812580c71d3389))

* Merge branch &#39;KennethEnevoldsen:main&#39; into main ([`9054c0a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/9054c0a1d449a68ebf1b62c5b23c8ad2a1f8e7f6))

* Corrected pre-commit errors ([`33c71ab`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/33c71ab8628d247b65b169777d00849f201841aa))

* CLI now accepts a list of languages as its input if none are passed the benchmark will be run on all languages ([`975418e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/975418ee3ef5263461152fa0b4244163f7791070))

* Merge pull request #24 from x-tabdeveloping/main

CLI is error tolerant now. ([`0bcfa98`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0bcfa981246abfe359bfe909f702ce0f886ed902))

* Fixed issue with mean calculation in DaLAJ. ([`b5d2f6b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b5d2f6ba27af3264f9e3e205c49344500e93356d))

* CLI made error tolerant prints NA for unobtainable benchmark results ([`69be78a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/69be78aa0f94c85c2302bcd29d5490def35cac79))


## v0.2.6 (2023-10-19)

### Fix

* fix: ran pyright ([`fed9f84`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/fed9f8456c3c8a293ff5214ec8d9bd22824018e8))

* fix: ran pyright and pre-commit ([`04dacf5`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/04dacf5dd5462f155d84c20edf714d13c4d756d8))

### Unknown

* Merge pull request #23 from KennethEnevoldsen/marton_changes

Fixes for #22 ([`0905a51`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0905a51ee1a19c2c18a3e5e673262143b61bd036))

* updated from cruft ([`7891730`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/789173026c4c60de91ecc68dee6cf8ba8bcab3e5))


## v0.2.5 (2023-10-19)

### Fix

* fix: update ci based on cruft template ([`32ea08d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/32ea08d427b2d5ae7d14d416cc38dcac0c956151))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/x-tabdeveloping/scandinavian-embedding-benchmark into marton_changes ([`3bf2688`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3bf2688c16c122c7d1e10b81ce4dbdb16d8fbfad))

* Added tabulate to dependencies ([`ca94dde`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ca94dde46b286abb9121170dedf793cda55ae728))

* Added main CLI for running benchmark. ([`fa92d35`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/fa92d35d408350639c536670cedbdb303015de1c))


## v0.2.4 (2023-09-25)

### Fix

* fix: Converted e5 embedding to the one specified by the docs ([`796a8dc`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/796a8dcf614f14023457bd3f5180bd5d72d41bf0))

* fix: ran ruff ([`172ad9a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/172ad9a3646a354567414d98b588782e6be5fba7))

### Style

* style: linting ([`44e3005`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/44e30055245914df854de90daf11a4f07d9aad31))

* style: linting ([`75a2edb`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/75a2edbe86acba0ee56ddf82bf2fa4829e963155))

### Unknown

* Merge pull request #21 from KennethEnevoldsen/KennethEnevoldsen/issue-Allow-custom-embeddings-based-on-the-task

Fix e5 models ([`af8cb17`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/af8cb17bdd5447565db035bd3de83904cd0eef30))

* Merge branch &#39;KennethEnevoldsen/issue-Allow-custom-embeddings-based-on-the-task&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark into KennethEnevoldsen/issue-Allow-custom-embeddings-based-on-the-task ([`76cf49e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/76cf49ee930f85355e00a360e65220a0d5030ba3))

* Allow custom embeddings based on the task
Fixes #18 ([`135e546`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/135e54640a75139b547166d45353ed415ea2ee43))


## v0.2.3 (2023-09-25)

### Ci

* ci: Remove dependabot ([`24ca03f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/24ca03f6ce8454cbc13ad4549feb9fa17deb0542))

### Fix

* fix: type hint ([`abeb2a7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/abeb2a7319c4facb19787352eb7a0e7a96dd32f7))

* fix: Added handling of long text for openai ([`20b087c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/20b087c888f405a50b1f09fb919ecdb3c95dc91c))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`9e9e624`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/9e9e624fcc2aaa4aceee098cbb04bdad795e2251))


## v0.2.2 (2023-09-25)

### Fix

* fix: ruff ([`51a0002`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/51a00020d3cb970b2f4065bab3a3a718a05d0ec7))

* fix: run on ucloud ([`aebaa4e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/aebaa4e59afc0620358f94174724e9e867c2ced0))

* fix: Add missing dependency ([`ae9571e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ae9571ea5da3e8ee364f4aabda2c5e0c2ee36569))

* fix: ran pyright ([`3f95827`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3f958275050a7bd7325c728921faf16417eaaa72))

* fix: Type hints ([`ccef58d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ccef58dbe5a5e23947ac7af68ef411713170597f))

* fix: ruff ([`3c2d922`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3c2d922a0ae5dcdf955c8071ad3127b3ca651995))

### Style

* style: linting ([`ab11cde`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ab11cde10cf70a96c2b9a66a5f1dda29aa1ee37b))

* style: linting ([`6467c42`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6467c4217cbdfea708dd95a69c3940b23c998b17))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/Scandinavian-Embedding-Benchmark ([`2e7bb7e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2e7bb7eb9cad1753a4d7cdfe633e1a62d4c6724c))

* Merge pull request #19 from KennethEnevoldsen/KennethEnevoldsen/issue-Add-OpenAI-embeddings

Add OpenAI embeddings ([`0ef4b34`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0ef4b34a4d999c7b3e71874726476cac4adfafba))

* merge ([`3612afa`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3612afa6f7d9b834f9a5cbaf9bb350990900aff4))

* Add OpenAI embeddings
Fixes #7 ([`0ab8418`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0ab84185a697420d61cc8e7e15095d856bd4470e))


## v0.2.1 (2023-09-13)

### Fix

* fix: Updated benchmark to handle errors ([`bdceba6`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/bdceba6caf188a61d863f3170bb4f693f54fe7a3))

* fix: Updated names and table sizes ([`7257d82`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7257d82ac954a0fb562880f36059b25429b425ae))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`8809741`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8809741183db99ece420803a2f16331e4277284c))


## v0.2.0 (2023-09-04)

### Build

* build: only import fairseq models if fairseq2 is installed ([`07dc22e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/07dc22ecf4cf51335a145736ab4c8cfda160391d))

* build: move fairseq and sonar to optional dependencies ([`0969f57`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0969f57f3dc6e042dee7f8be3b046e2fc8f8dd8e))

* build: change to pypi version of fairseq2 ([`58a62c9`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/58a62c93bb8010dd64c418ad39fc4d5ab403f698))

* build: add sonar and fairseq2 dependencies ([`03d3971`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/03d3971584295a8b36c17301999c0753d4c45d73))

### Documentation

* docs: human translated != machine translated ([`0d8f86a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/0d8f86a688fd164eb88fff170fd2f076f40c535f))

* docs: Updated docs ([`423fd4e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/423fd4ebf25a21dd9f24d1d5612dd923401f1664))

* docs: removed requirements for social card ([`fb37b26`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/fb37b26d4e33d6f76a4b93591726a7459a77a3fb))

### Feature

* feat: add sonar model ([`da45fca`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/da45fca623d1146a0ef0d1cc2e92b3e330df0045))

* feat: Added SweFAQ ([`c56fdb7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c56fdb7439b5eff0542521e1d1faab12936107f5))

### Fix

* fix: Overwriting task creator ([`5da1aed`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/5da1aeda4500b3bf1e13b6758f73bcd5c5f90e71))

* fix: Updated such that sonar model will get registrered, but not run ([`28d151f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/28d151ff1864ea799c1ee04076730d85d6332827))

* fix: Updated pyright ([`6fe494f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6fe494fadcb9a63fb3e78eef0a5b5fdda4f33f53))

* fix: Updated task name ([`f81266a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f81266ac715639561606c0f9fa27aefc9d8e752a))

* fix: make sonar return numpy.ndarray ([`b1db630`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b1db63081d2a5a7dd2f6ec493c3824a0fc8b7bf2))

* fix: unique names for sonar models ([`d7f353d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d7f353d5a7c79e1111aefaa7451bf5ec46b2f325))

* fix: add model imports to init ([`267fb0b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/267fb0bcb1b6886c4b72a556df250ec6230d08e5))

* fix: change folder structure, add sonar model per language ([`94a293c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/94a293c7f6be548bed6baed136185ddb844458d6))

* fix: add languages to sonar model ([`f1d81a1`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f1d81a1534967f56fffdd64097fea9d49d425a1f))

* fix: Added beir requirement for retrieval tasks ([`2a614ba`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2a614ba30afcca3eb4fb950697d4793798868f5e))

* fix: Updated metadata for SweFAQ ([`cc2d6c3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/cc2d6c324e2825dea04d67c36e3538865c78dc2a))

* fix: Updated dataset name ([`48db87e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/48db87e9bdbd9a4c6fda244cafcde00454442405))

### Style

* style: lint ([`d9790ad`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d9790ad2336da3a6d307fa332e1b2f8ac98bd22a))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`9f0b422`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/9f0b4229c52ab2b60e6d7ace89e681d5e30fda78))

* Merge pull request #14 from HLasse/sonar

feat: add sonar model ([`082c157`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/082c1579789a3cb5887b7e7746d6fb18b25a21d5))

* Merge pull request #15 from KennethEnevoldsen/dependabot/pip/pyright-1.1.324

deps:(deps-dev): bump pyright from 1.1.323 to 1.1.324 ([`7cfd512`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7cfd512fa70696c5599ecfb597e3c39357b1e59d))

* deps:(deps-dev): bump pyright from 1.1.323 to 1.1.324

Bumps [pyright](https://github.com/RobertCraigie/pyright-python) from 1.1.323 to 1.1.324.
- [Release notes](https://github.com/RobertCraigie/pyright-python/releases)
- [Commits](https://github.com/RobertCraigie/pyright-python/compare/v1.1.323...v1.1.324)

---
updated-dependencies:
- dependency-name: pyright
  dependency-type: direct:production
  update-type: version-update:semver-patch
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt; ([`7774c60`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7774c60efc2d201940f8683086d89d815edf0411))

* Merge pull request #13 from KennethEnevoldsen/dependabot/pip/pyright-1.1.323

deps:(deps-dev): bump pyright from 1.1.322 to 1.1.323 ([`48142d3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/48142d3cc4b8779060ba81e4eed3cf0168748f64))

* deps:(deps-dev): bump pyright from 1.1.322 to 1.1.323

Bumps [pyright](https://github.com/RobertCraigie/pyright-python) from 1.1.322 to 1.1.323.
- [Release notes](https://github.com/RobertCraigie/pyright-python/releases)
- [Commits](https://github.com/RobertCraigie/pyright-python/compare/v1.1.322...v1.1.323)

---
updated-dependencies:
- dependency-name: pyright
  dependency-type: direct:production
  update-type: version-update:semver-patch
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt; ([`99e112d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/99e112da63ebcee8e8acccfbd954b22fe12fb657))

* Merge pull request #12 from KennethEnevoldsen/dependabot/pip/pyright-1.1.322

deps:(deps-dev): bump pyright from 1.1.320 to 1.1.322 ([`8086b5a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8086b5af2e099bd139bdcc1cf42c3a6a8bee7b62))

* deps:(deps-dev): bump pyright from 1.1.320 to 1.1.322

Bumps [pyright](https://github.com/RobertCraigie/pyright-python) from 1.1.320 to 1.1.322.
- [Release notes](https://github.com/RobertCraigie/pyright-python/releases)
- [Commits](https://github.com/RobertCraigie/pyright-python/compare/v1.1.320...v1.1.322)

---
updated-dependencies:
- dependency-name: pyright
  dependency-type: direct:production
  update-type: version-update:semver-patch
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt; ([`2bd7736`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2bd773665cae37e2597bf3e1f3a165e7737af7bd))

* Merge pull request #11 from KennethEnevoldsen/dependabot/pip/pyright-1.1.320

deps:(deps-dev): bump pyright from 1.1.318 to 1.1.320 ([`be26f91`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/be26f9125b960764a2ba7af5113a257a30fbb227))

* deps:(deps-dev): bump pyright from 1.1.318 to 1.1.320

Bumps [pyright](https://github.com/RobertCraigie/pyright-python) from 1.1.318 to 1.1.320.
- [Release notes](https://github.com/RobertCraigie/pyright-python/releases)
- [Commits](https://github.com/RobertCraigie/pyright-python/compare/v1.1.318...v1.1.320)

---
updated-dependencies:
- dependency-name: pyright
  dependency-type: direct:production
  update-type: version-update:semver-patch
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt; ([`2032c84`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2032c845424c34ed883c108d4a1924a2fd4f04f4))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`a50ee2c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a50ee2cb583d64c9ac76838639f6cc0369b27214))


## v0.1.5 (2023-08-01)

### Ci

* ci: fixed version pointer ([`4af85d1`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4af85d11af1b22e41ee53119235a5184c80ce951))

### Documentation

* docs: test social card ([`2460524`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2460524a96235305e4462184ed9a66e3194b0664))

* docs: re-add social card ([`ff46543`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ff465430d5c807cb4050b6364f6f7abe7ece389b))

### Fix

* fix: pyproject.toml version reference ([`5968bd7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/5968bd7df7dbba3b84373de397e277ddb966c1f7))

### Unknown

* fix:empty commit for ci ([`db43d77`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/db43d7774c539f0803916d89cc1ad55f36345ad6))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`68ed93d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/68ed93db75bccfabebef3c24f36f9c603786106d))


## v0.1.4 (2023-08-01)

### Documentation

* docs: Updated links in badge ([`8e3ab16`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8e3ab168b518e095d6f8cd8bf5d8afb2f82de0ba))

### Fix

* fix: ci: removed outdated variables from pyproject.toml ([`b43edd8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b43edd86701704d25323feb61b5140b5f1059217))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`84244d2`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/84244d2117ddf11da2eaf37c4bab1dbef03807d2))


## v0.1.3 (2023-08-01)

### Ci

* ci: Added permissions ([`51de97a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/51de97a2af8d43318d6c0e08a51614459fa858b5))

### Fix

* fix: rerun ci ([`94362a9`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/94362a97999b17516bdad2bb241675c038ea7350))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`7301671`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/73016713ca2b4c253f8211e3ee4db5351bd27fa9))


## v0.1.2 (2023-08-01)

### Ci

* ci: re-added gh token ([`a280ee6`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a280ee6315c43c8603363d5408065747d0767644))

* ci: Updated release ci to use pypi trusted publishing ([`e5cec02`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e5cec024bce8e907d1edbbf74509360d359e732c))

### Fix

* fix: empty commit ([`73db296`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/73db296636c7cc929cf8bb8e4fde2073241a0873))

* fix: Empty commit ([`cc1dfd7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/cc1dfd73c2f011d8ece541effa112b52362e1cc6))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`66bfdf0`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/66bfdf0e9107f00758ae14377885e7fe7e2d9220))


## v0.1.1 (2023-08-01)

### Fix

* fix: Added documentation to all outward facing functions ([`ac96fc1`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ac96fc15fadb7e96306b5479c053f9fb229b2019))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`1464b68`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/1464b6887dc0c421a8a9c74ae5cf3dd5f9e82920))


## v0.1.0 (2023-08-01)

### Build

* build: Added python 3.11 ([`2084983`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2084983b38e2cbf09499284acf2a7fabcdb6eee1))

* build: Added missing requirement for docs ([`c9d9090`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c9d909094d9b2409b5c644c28eb3be1a566e90f9))

### Ci

* ci: updated to latest version of release ([`bc6394f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/bc6394f1bda7552432aa76e258b0d99da5c5a39a))

* ci: fix release action ([`419dc23`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/419dc23726c4c25a051e90a52a34b5543099e958))

* ci: Fix inv script to avoid issues with pyright ([`63de866`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/63de866f13ace329d906141a0a4c6651271f5e05))

* ci: fix url ([`669b258`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/669b2580e1c3e57b0966e11ee7633a030b21ecff))

* ci: Updated docs build to mkdocs ([`9cae854`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/9cae85401de9d4fe6f34a212004d28af03be8ef7))

* ci: Removed windows from CI ([`4d037c2`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4d037c249f0a9bd445c57eca7c50a59f1e43371b))

* ci: Added macOS and windows OS ([`e4084eb`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e4084eb08a27b358f18245d4eb9d6a373e7578b9))

### Documentation

* docs: Added missing dependencies for docs ([`ba4fc76`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ba4fc7652f1c2c3a417043410975c42eecb8abfb))

* docs: retrying to get the social card to work ([`6403f7d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6403f7dda7154c075da5afee5f04dd5bb56eb44e))

* docs: Minor adjustment ([`4fc6881`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4fc6881c96a3a7cc86212429077c450298ce20b6))

* docs: Added social card ([`1b692f6`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/1b692f61151924aaef2b25b17c19efb338a853a2))

* docs: Updated references when sharing table ([`7e45a7a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/7e45a7aac9dede71ce0f0dfb1c6e4602d19d9c34))

* docs: Updated table with new models ([`7083239`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/70832393d4202844d53957efb221787e97a31a53))

* docs: minor updates to index page ([`8a883c4`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8a883c4255864c9ba8d567b8e3fa5e42236642e0))

* docs: testing iframes ([`d415cb8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d415cb835be914710407e3a533557a8158c04af2))

* docs: Updated page tile ([`ce6ed9a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ce6ed9a1dd09eaa4e1d8fefe5b5d12310f3d1da0))

* docs: Updated tables ([`dcceccb`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/dcceccb81580e934f8dd79d5eb94ca792c5ee0be))

* docs: fixed heights ([`6750010`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/67500100662f2d85268a9bf437a47b8bb36ff9a0))

* docs: Updated tables ([`f62e506`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f62e50635a31ddb6fb1749b21751450a7f671511))

* docs: added notice regarding tests ([`b356e62`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/b356e6228d551d929add995253ada42ff24d0c2c))

* docs: Updated description ([`67f0200`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/67f0200b5f763df53938503b9ef30fcc9e1c5287))

* docs: Updated tables ([`aa9d7d0`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/aa9d7d05fb1015a0d2396dc2f969284fd2485425))

* docs: Minor changes to headings ([`5fda1ff`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/5fda1ff740028d462aae4629d22d320cbe4725f8))

* docs: added logo ([`56d26f8`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/56d26f80de1153b538da892087f99dd28d9b7650))

* docs: Added logo ([`79d1a17`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/79d1a17c186662db6dfd924b434987153250b9d8))

* docs: Updated readme ([`fe57c81`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/fe57c81f35f5f0d415badb1387cc7525a2cd3e0d))

* docs: Updated the documentation ([`4875b62`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4875b62ef6eb1c4f6d8adc840b081aba2cfd2655))

* docs: Added link and removed nav. on main page ([`1f284db`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/1f284dbbc946b602f8c2f95aa56eef625cdb667d))

* docs: Updated docs workflow to use mkdocs ([`2b7a6b4`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2b7a6b498c802f7031d1ed47fa677861c0f8855b))

### Feature

* feat: add multilingual sentence transformer ([`6e4ae7a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6e4ae7afb9829984d62798d2e5657b0d88c93fb8))

* feat: Bumped version

Still awaiting PR: https://github.com/embeddings-benchmark/mteb/pull/128 ([`5240256`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/5240256ac69e54748cd6b855b82d4df404351bd1))

### Fix

* fix: removed hotfix for MTEB ([`c2e08b3`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c2e08b3f253306f995cfa5eb6e5cd31bed7b9504))

* fix: Added language codes for English models ([`8615d3d`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8615d3d6a0b962500be5d1ffdd6e708e06e59ea6))

* fix: Added ignore for static type check ([`caca3d7`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/caca3d739dda2bebbd8c81f12c42c6b1f7e756ba))

* fix: temporary hotfix while waiting for mteb merge

waiting for https://github.com/embeddings-benchmark/mteb/pull/128 ([`932b847`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/932b847469e6ce8c5357d583a5b59d7f36457daa))

* fix: Updated string handling for model references ([`39a1060`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/39a106010774730660dca627526825372b858a87))

* fix: Correctly sorted values in tables ([`e69b4bb`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e69b4bb9bbeb0bfeadc546077ec57fd8f2fc907e))

* fix: Added full benchmark ([`e3d179c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e3d179c878f38b958e3559643b278a67dfc3b378))

* fix: Updated type hint ([`963b099`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/963b0993da4a4bdef85fa95b3f1eb2161d79c02b))

* fix: Updated language codes to exclude &#34;no&#34; ([`c2c14ed`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c2c14ed41c8b1dc61a0b6c4a0bf931a7821ce042))

* fix: filtered logging and added progress bar ([`ada3dae`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ada3daeeaa123239dff8fefcb9dedb058d6aa981))

* fix: Added time to task errors as well ([`4c79425`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4c794257c346dea5b388e82db60189281054cfe1))

* fix: Updated task versioning ([`2bc628f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/2bc628f54442277c84be96b94b8ba9deb009690b))

* fix: Added all models ([`8b69593`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/8b6959328c1d7c5a5cb695d830f31d7b81b552ae))

* fix: Added cache and error handling ([`e400cfa`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/e400cfa1402265ee7ace89770cde02d25a89828e))

* fix: Test loads and fail ([`d66b1a6`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d66b1a6d5b64e132b145606d49bcb4dbb7756105))

* fix: removed cli for now ([`ab03b7b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ab03b7bfb279e3835712c7b809832fa673617436))

### Style

* style: Only enforce pyright on src folder ([`0692145`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/069214509babf137c565dfbc403b45e47a3b3999))

### Unknown

* tests: Remove downloaded tasks from test ([`29d5e50`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/29d5e508ced2b40d3719408dc228da05eea8f892))

* tests: Set custom cache dir for tests ([`188cd46`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/188cd460aba621e0f7db169714b7ae3df03fc64e))

* tests: Fix task reference ([`ed6dc97`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/ed6dc9753df41c477a09aa0fc1e963ea0c763bb3))

* tests: Skip tests with large downloads ([`d2d9c64`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d2d9c64962703dc5114f8b9fe47f87117eb5c33b))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`21ffc06`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/21ffc064719bdedbbb08ee6d8adf8f0dc6750b66))

* Merge pull request #5 from HLasse/patch-2

feat: add multilingual sentence transformer ([`15b8045`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/15b8045e9959edb521f86ce527773b767be6cda7))

* Merge pull request #4 from HLasse/patch-1

docs: minor updates to index page ([`c3d7b69`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c3d7b69922617a149624982477dff168d59c0812))

* Merge branch &#39;main&#39; of https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark ([`a1b8182`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/a1b8182b3be0148f83e90fb3381511746dc10520))

* v0.0.0 ([`6703d6c`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/6703d6c2466d28cd51a85d2f12c78be909328507))

* Merge pull request #2 from KennethEnevoldsen/dependabot/pip/pyright-1.1.318

deps:(deps-dev): bump pyright from 1.1.305 to 1.1.318 ([`d5f448e`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/d5f448e61cfe53d5d776b7c1a0c1cdd79b472e93))

* Merge pull request #1 from KennethEnevoldsen/dependabot/pip/invoke-2.2.0

deps:(deps-dev): bump invoke from 2.1.1 to 2.2.0 ([`c463449`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c4634495afdafc431f182c98f28cf335be5dee9e))

* deps:(deps-dev): bump pyright from 1.1.305 to 1.1.318

Bumps [pyright](https://github.com/RobertCraigie/pyright-python) from 1.1.305 to 1.1.318.
- [Release notes](https://github.com/RobertCraigie/pyright-python/releases)
- [Commits](https://github.com/RobertCraigie/pyright-python/compare/v1.1.305...v1.1.318)

---
updated-dependencies:
- dependency-name: pyright
  dependency-type: direct:production
  update-type: version-update:semver-patch
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt; ([`3b28588`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/3b285889472504361b3a9af0b3f9218077766442))

* deps:(deps-dev): bump invoke from 2.1.1 to 2.2.0

Bumps [invoke](https://github.com/pyinvoke/invoke) from 2.1.1 to 2.2.0.
- [Commits](https://github.com/pyinvoke/invoke/compare/2.1.1...2.2.0)

---
updated-dependencies:
- dependency-name: invoke
  dependency-type: direct:production
  update-type: version-update:semver-minor
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt; ([`c114c6b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/c114c6b8a91a44c65e1123a772f5de35b2b0dade))

* Added all remaining tasks ([`5e7511f`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/5e7511ff641c8776714f037cdc5066b9751b2e19))

* Added mkdocs ([`bd6cb42`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/bd6cb42e635f990b3f80773d427c88828af1ffb2))

* Added in all models ([`f7f4d1b`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/f7f4d1b658e580d52f9fb65f92568c6d5572d5ea))

* all tests pass ([`7500552`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/75005522ed1f951e36da43b532b05ee9f6da78bd))

* remove unused files ([`4f5a93a`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/4f5a93ac28b796bfea78431a721c5fc7ac6828c4))

* initial commit ([`efaceb4`](https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/commit/efaceb4c0a9f522a4f036234a0b2a66cb9e252f7))
