'use strict';
const $=(selector,root=document)=>root.querySelector(selector);
const make=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=String(text);return n;};
const button=(text,action,cls='secondary')=>{const n=make('button',cls,text);n.type='button';n.onclick=action;return n;};
const link=(text,href,cls='text-button')=>{const n=make('a',cls,text);n.href=href;return n;};
const stamp=value=>value?new Date(value).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}):'—';
const labels={queued:'已排队',preparing:'准备实验',running:'运行中',rendering:'视觉渲染中',completed:'已完成',failed:'失败',blocked:'无法执行',unavailable:'不可用',interrupted:'已中断',cancelled:'已取消',ready:'就绪',missing:'缺失',partial:'部分结果',error:'错误',not_applicable:'不适用',ok:'已记录',pending:'待执行'};
const badge=value=>make('span','badge '+(value||''),labels[value]||value||'未记录');
const metricLabels={simulated_seconds:'仿真时长（秒）',goal_reached:'是否到达',goal_distance_m:'距终点（米）',fallen:'是否跌倒',travel_distance_m:'行进距离（米）'};
const metricValue=value=>value===null||value===undefined?'未测得':typeof value==='boolean'?value?'是':'否':typeof value==='number'&&Number.isFinite(value)?value.toLocaleString('zh-CN',{maximumFractionDigits:3}):typeof value==='object'?JSON.stringify(value):String(value);
const state={rows:[],id:null,record:null,revision:0,signature:null,evidenceSignature:null,evidenceId:null,dispose:()=>{},task:new URLSearchParams(location.search).get('task')};
async function api(path){const r=await fetch(path);if(!r.ok){let message=await r.text();try{message=JSON.parse(message).detail||message;}catch{}throw Error(typeof message==='string'?message:JSON.stringify(message));}return r.json();}
function notice(message=''){$('#notice').textContent=message;$('#notice').hidden=!message;}
function artifact(path,id=state.id){return '/api/rollouts/'+encodeURIComponent(id)+'/artifact?path='+encodeURIComponent(path);}
function fieldPanel(title,data){const panel=make('section','panel summary-panel');panel.append(make('h3','',title));if(data==null||typeof data==='object'&&!Object.keys(data).length){panel.append(make('p','muted small','尚无归档记录。'));return panel;}const pre=make('pre','rollout-json',typeof data==='string'?data:JSON.stringify(data,null,2));panel.append(pre);return panel;}
function details(title,data){const n=make('details','advanced'),pre=make('pre','rollout-json',JSON.stringify(data,null,2));n.append(make('summary','',title),pre);return n;}
function preview(value){
  if(typeof value==='string'&&value.startsWith('data:image/')&&value.includes(';base64,'))return `[内联图像长字段已折叠：${value.length.toLocaleString('zh-CN')} 字符；下载原始记录可查看完整内容。]`;
  if(typeof value==='string')return value.length>4096?value.slice(0,512)+'\n… 长字段已折叠；下载原始记录可查看完整内容。':value;
  if(Array.isArray(value))return value.map(preview);
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,preview(item)]));
  return value;
}
function archivedDetails(title,path,id){
  const root=make('details','advanced rollout-artifact'),summary=make('summary','',title),body=make('div','rollout-artifact-body');
  root.append(summary,body);let loaded=false;
  root.addEventListener('toggle',async()=>{
    if(!root.open||loaded)return;loaded=true;body.replaceChildren(make('p','muted small','正在读取归档记录…'));
    try{
      const response=await fetch(artifact(path,id));if(!response.ok)throw Error('读取归档失败：'+response.status);
      const raw=await response.text();let value;try{value=JSON.parse(raw);}catch{value=raw;}
      const download=link('下载原始记录',artifact(path,id),'text-button small');download.download=path.split('/').at(-1);
      body.replaceChildren(download,make('pre','rollout-json',typeof value==='string'?preview(value):JSON.stringify(preview(value),null,2)));
    }catch(error){loaded=false;body.replaceChildren(make('p','error',error.message));}
  });return root;
}
function renderReference(evidence,id){
  const target=$('#rollout-reference');if(!target)return;
  const signature=JSON.stringify(evidence.reference||{});if(target.dataset.signature===signature)return;target.dataset.signature=signature;
  const windowSeconds=state.record?.result?.render_window_s;
  const anchorNote=windowSeconds?`首段使用生成的参考图，每 ${windowSeconds} 秒用上一段末帧接续。`:'这里展示实验启动时使用的参考图。';
  target.replaceChildren(make('h3','','参考图与生成依据'),make('p','muted small','图 1 提供布局与相机视角，图 2 提供材质、光照和纹理。'+anchorNote));
  const reference=evidence.reference||{},grid=make('div','rollout-reference-grid');
  const definitions=[['proxy','图 1 · 物理场景 proxy','几何与视角'],['appearance','图 2 · 原始场景参考帧','外观依据'],['generated','生成的初始参考图','gpt-image-2'],['used','首段使用的参考图','输入渲染器的尺寸']];
  for(const[key,title,caption]of definitions){
    const row=reference[key]||{};
    if(key==='used'&&!row.path)continue;
    const card=make('figure','rollout-reference-card'),frame=make('div','rollout-reference-frame');
    if(row.path){
      const image=make('img');image.src=artifact(row.path,id);image.alt=title;image.loading='lazy';
      image.addEventListener('error',()=>{frame.replaceChildren(make('p','muted small','图像无法读取；原始归档仍可下载。'));});
      const open=link('',artifact(row.path,id),'');open.target='_blank';open.rel='noopener';open.setAttribute('aria-label','查看原图：'+title);open.append(image);frame.append(open);
    }else frame.append(make('p','muted small',row.reason||'尚未生成或未留存此图像。'));
    const description=make('figcaption');description.append(make('strong','',title),make('small','muted',caption));
    if(row.path){const download=link('下载原图',artifact(row.path,id),'text-button small');download.download=row.path.split('/').at(-1);description.append(download);}
    card.append(frame,description);grid.append(card);
  }
  target.append(grid);
  for(const[key,title]of [['request_path','图像编辑提示词与请求'],['receipt_path','图像编辑调用回执'],['provenance_path','原始参考帧来源']]){
    if(reference[key])target.append(archivedDetails(title,reference[key],id));
  }
  if(windowSeconds)target.append(archivedDetails('渲染分段与参考帧','render_chunks.json',id));
}
function renderExecution(evidence,id){
  const target=$('#rollout-execution');if(!target)return;
  const events=evidence.events||[],invocations=evidence.invocations||[];
  target.replaceChildren(make('h3','','评测 agent 执行过程'));
  const calls=events.filter(row=>row.kind==='call').length,tools=events.filter(row=>row.kind==='tool').length;
  target.append(make('p','muted small',events.length?`${calls} 次模型调用 · ${tools} 次工具执行。输入、返回和日志按需读取，原始记录完整保留。`:'此实验尚无可读取的评测 agent 调用记录。'));
  for(const invocation of invocations){
    const row=make('div','rollout-invocation');row.append(make('strong','',invocation.role||'评测 agent'),badge(invocation.status));
    if(invocation.error)row.append(make('p','error small',invocation.error));
    if(invocation.context_path)row.append(archivedDetails('本次输入与系统提示词',invocation.context_path,id));
    if(invocation.result_path)row.append(archivedDetails('Agent 收尾输出',invocation.result_path,id));
    target.append(row);
  }
  const timeline=make('div','execution-timeline rollout-execution-timeline');
  for(const[eventIndex,event]of events.entries()){
    const entry=make('details','rollout-event'),summary=make('summary'),label=make('span','rollout-event-label'),kind=event.kind==='tool'?'工具':'模型调用';
    label.append(make('strong','',`${kind} · ${event.name||event.model||event.role||event.id}`),make('small','muted',`${stamp(event.created_at)}${event.usage?.total_tokens!=null?' · '+Number(event.usage.total_tokens).toLocaleString('zh-CN')+' tokens':''}`));
    summary.append(make('span','call-number',String(eventIndex+1).padStart(2,'0')),label,badge(event.status));
    const body=make('div','rollout-event-body');
    if(event.arguments!==undefined){let args=event.arguments;try{if(typeof args==='string')args=JSON.parse(args);}catch{}body.append(details('工具输入',args));}
    if(event.result!==undefined)body.append(details('工具返回摘要',preview(event.result)));
    for(const[key,title]of [['request_path','模型请求'],['response_path','模型返回'],['meta_path','调用信息'],['result_path','完整工具返回']]){
      if(event[key])body.append(archivedDetails(title,event[key],id));
    }
    for(const log of event.logs||[]){if(log.path)body.append(archivedDetails(log.label||'运行日志',log.path,id));}
    if(!body.childNodes.length)body.append(make('p','muted small','此事件尚无详细归档。'));
    entry.append(summary,body);timeline.append(entry);
  }
  target.append(timeline);
}
async function loadEvidence(id,revision){
  try{
    const evidence=await api('/api/rollouts/'+encodeURIComponent(id)+'/evidence');
    if(revision!==state.revision||id!==state.id)return;
    const signature=JSON.stringify(evidence);if(state.evidenceId===id&&state.evidenceSignature===signature&&$('#rollout-execution')?.dataset.loaded)return;
    state.evidenceId=id;state.evidenceSignature=signature;
    const instruction=$('#rollout-instruction');if(instruction){instruction.replaceChildren(make('h3','','原始任务指令'),make('p','text-block rollout-original-instruction',evidence.instruction||'原任务未附加文字指令。'),make('p','muted small','此处保留威胁场景生成 task 的原始用户指令，评测阶段沿用同一任务意图。'));if(evidence.instruction_matches_archive===false){instruction.append(make('p','policy-note','当前任务记录与执行时归档的用户指令不同；历史输入保持原样，可展开核对。'),details('执行时归档的用户指令',evidence.archived_instruction));}}
    renderReference(evidence,id);renderExecution(evidence,id);if($('#rollout-execution'))$('#rollout-execution').dataset.loaded='true';
  }catch(error){
    if(revision!==state.revision||id!==state.id)return;
    for(const selector of ['#rollout-reference','#rollout-execution']){const target=$(selector);if(target){target.replaceChildren(make('h3','',selector.includes('reference')?'参考图与生成依据':'评测 agent 执行过程'),make('p','muted small','归档证据暂不可读取：'+error.message));}}
    const instruction=$('#rollout-instruction');if(instruction)instruction.replaceChildren(make('h3','','原始任务指令'),make('p','muted small','原始任务指令暂不可读取，请稍后刷新。'));
  }
}
function renderList(){const target=$('#records'),query=$('#search').value.toLowerCase(),rows=state.rows.filter(row=>(!state.task||row.task_id===state.task)&&JSON.stringify([row.title,row.id,row.task_id]).toLowerCase().includes(query));target.replaceChildren();$('#rollout-count').textContent=String(state.rows.length);for(const row of rows){const n=button('',()=>select(row.id),'record'+(row.id===state.id?' active':''));n.append(make('div','record-name',row.title||row.id));const meta=make('div','record-meta');meta.append(make('span','',stamp(row.created_at)),badge(row.status));n.append(meta);target.append(n);}if(!rows.length)target.append(make('p','record-empty',state.task?'此场景暂无评测记录':'尚无 rollout 记录'));}
function home(){state.dispose();state.signature=null;state.evidenceSignature=null;state.evidenceId=null;const target=$('#content');target.replaceChildren();const head=make('div','page-heading'),text=make('div');text.append(make('span','eyebrow','ROLLOUT EXPERIMENT REVIEW'),make('h1','','查看实验，而后沉淀经验'),make('p','','独立评测 agent 按原始任务指令，在已交付场景中配置具身模型，保留执行过程、参考图和真实运行结果。'));head.append(text);target.append(head);const panel=make('section','panel summary-panel');panel.append(make('h3','',state.task?'此场景尚无 rollout':'尚无可查看的 rollout'),make('p','muted','完成的场景保持独立。评测准备、模型加载或渲染失败也会保留记录；没有真实运行证据时不显示成功结论。'),link(state.task?'返回威胁场景':'打开 safeRSI',state.task?'/safe#task/'+encodeURIComponent(state.task):'/safe'));target.append(panel);if(state.task)target.append(link('查看全部 rollout 历史','/rollouts'));}
async function select(id,preserve=false){const revision=++state.revision;state.id=id;document.body.classList.remove('menu-open');history.replaceState(null,'','#rollout/'+encodeURIComponent(id));renderList();if(!preserve)$('#content').replaceChildren(make('div','loading','正在读取实验记录…'));try{const record=await api('/api/rollouts/'+encodeURIComponent(id));if(revision!==state.revision)return;state.record=record;const signature=JSON.stringify(record);if(signature!==state.signature){state.signature=signature;render(record);}await loadEvidence(record.id,revision);notice();}catch(error){if(revision===state.revision)notice(error.message);}}
function renderVideos(target,record) {
  const controls=make('div'),grid=make('div','scene-grid rollout-video-grid');
  target.append(make('p','policy-note','导航在决策时读取神经渲染画面。同一时刻的 Warp 首视角可用于核对障碍物位置和能见度。'),controls,grid);
  const tracks=[
    ['first_person','第一人称 · 导航画面','NEURAL RGB','尚无神经渲染后的第一人称视频'],
    ['first_person_proxy','第一人称 · 原始画面','WARP RGB','尚无原始第一人称视频'],
    ['third_person','第三人称仿真','WARP RGB','尚无第三人称仿真视频']
  ];
  state.dispose=playback(controls,tracks.map(track=>mediaCard(grid,record,...track)).filter(Boolean));
  if(record.receipt_path)target.append(archivedDetails('控制器归档回执',record.receipt_path,record.id));
}
function mediaCard(target,record,key,title,subtitle,missing){const entry=record.videos?.[key],value=typeof entry==='string'?{path:entry}:entry||{},card=make('section','scene-card rollout-video-card'),head=make('div','scene-heading'),frame=make('div','scene-frame'),bottom=make('div','scene-bottom');head.append(make('span','',title),make('small','',subtitle));card.append(head,frame,bottom);target.append(card);if(!value.path){const empty=make('div','scene-empty');empty.append(make('span','','◇'),make('div','',missing));frame.append(empty);bottom.append(badge(value.status||'unavailable'));if(value.reason||value.error)card.append(make('p','rollout-media-note error',value.reason||value.error));return null;}const video=make('video');video.controls=true;video.playsInline=true;video.preload='metadata';video.muted=true;video.src=artifact(value.path,record.id);video.setAttribute('aria-label',title);video.addEventListener('error',()=>{if(!frame.querySelector('.video-error'))frame.append(make('div','video-error','视频加载失败；可下载归档文件并检查实验记录。'));});frame.append(video);bottom.append(badge(value.status||(record.status==='completed'?'completed':'partial')),link('下载视频',artifact(value.path,record.id),''));if(value.original_path&&value.original_path!==value.path)bottom.append(link('原始 MP4',artifact(value.original_path,record.id),''));if(value.frame_count||value.frames||value.fps)bottom.append(make('small','',`${value.frame_count||value.frames||'—'} 帧 · ${value.fps||'—'} FPS`));return video;}
function playback(target,videos){const root=make('div','playback-toolbar'),play=button('▶ 同步播放',toggle),reset=button('回到起点',()=>seek(0),'text-button'),range=make('input'),clock=make('span','playback-time','0.00 s');range.type='range';range.min=0;range.max=0;range.step=.01;range.value=0;range.setAttribute('aria-label','Rollout 播放时间');root.append(play,reset,range,clock,make('small','','按归档视频时间同步播放；实际仿真采样与时间戳保存在实验记录中。'));target.append(root);let duration=0,playing=false,raf=0,guard=false,disposed=false;function update(t){range.value=t;clock.textContent=t.toFixed(2)+' / '+duration.toFixed(2)+' s';}function ready(){duration=Math.max(0,...videos.map(v=>Number.isFinite(v.duration)?v.duration:0));range.max=duration;play.disabled=!duration;reset.disabled=!duration;}function pause(){playing=false;play.textContent='▶ 同步播放';guard=true;videos.forEach(v=>v.pause());guard=false;cancelAnimationFrame(raf);}function seek(t){guard=true;videos.forEach(v=>{if(v.readyState&&Math.abs(v.currentTime-t)>.025)v.currentTime=Math.min(t,Number.isFinite(v.duration)?v.duration:t);});guard=false;update(t);}function tick(){if(disposed||!playing)return;const video=videos.find(v=>!v.paused&&!v.ended);if(!video){pause();return;}update(video.currentTime);raf=requestAnimationFrame(tick);}async function start(){if(!duration)return;if(Number(range.value)>=duration-.01)seek(0);playing=true;play.textContent='Ⅱ 暂停';guard=true;await Promise.allSettled(videos.map(v=>v.play()));guard=false;if(!disposed&&playing)raf=requestAnimationFrame(tick);}function toggle(){playing?pause():start();}range.oninput=()=>seek(Number(range.value));videos.forEach(v=>{v.addEventListener('loadedmetadata',ready);v.addEventListener('play',()=>{if(!guard&&!playing)start();});v.addEventListener('pause',()=>{if(!guard&&playing&&!v.ended)pause();});v.addEventListener('seeking',()=>{if(!guard)seek(v.currentTime);});});const visibility=()=>{if(document.hidden)pause();};document.addEventListener('visibilitychange',visibility);ready();return()=>{disposed=true;pause();document.removeEventListener('visibilitychange',visibility);videos.forEach(v=>{v.removeAttribute('src');v.load();});};}
function render(record){state.dispose();const target=$('#content');target.replaceChildren();const head=make('div','page-heading'),text=make('div'),actions=make('div','actions');text.append(make('span','eyebrow','ROLLOUT EXPERIMENT REVIEW'),make('h1','',record.title||record.id),make('p','',`${stamp(record.created_at)} · ${record.id}`));const taskId=record.task_id||record.scene_source?.task_id;if(taskId){actions.append(link('自由浏览威胁场景 ↗','/safe#task/'+encodeURIComponent(taskId)+'/model','secondary'),link('场景生成与 RSI','/safe#task/'+encodeURIComponent(taskId)+'/rsi','secondary'));}head.append(text,actions);target.append(head);const meta=make('div','info-chips'),status=make('span','info-chip');status.append(make('span','','实验状态'),badge(record.status));meta.append(status);if(record.phase){const phase=make('span','info-chip');phase.append(make('span','','当前步骤'),make('b','',record.phase));meta.append(phase);}target.append(meta);if(state.task)target.append(link('查看全部 rollout 历史','/rollouts'));if(record.error){const panel=make('section','panel summary-panel');panel.append(make('h3','','实验未完整完成'),make('p','error',typeof record.error==='string'?record.error:JSON.stringify(record.error)));target.append(panel);}if(record.agent_error){const panel=make('section','panel summary-panel');panel.append(make('h3','','评测 agent 收尾'),make('p','muted small',record.status==='completed'?'实际仿真结果与视频已归档；评测 agent 未完成收尾。':'评测 agent 未完成收尾，实际运行结果以归档证据为准。'),make('p','error',typeof record.agent_error==='string'?record.agent_error:JSON.stringify(record.agent_error)));target.append(panel);}const instruction=make('section','panel summary-panel rollout-instruction');instruction.id='rollout-instruction';instruction.append(make('h3','','原始任务指令'),make('p','muted small','正在读取与威胁场景生成阶段一致的用户指令…'));target.append(instruction);renderVideos(target,record);const reference=make('section','panel summary-panel');reference.id='rollout-reference';reference.append(make('h3','','参考图与生成依据'),make('p','muted small','正在读取归档参考图…'));target.append(reference);const result=make('section','panel summary-panel');result.append(make('h3','','真实运行结果'));const metrics=record.metrics||record.result?.metrics||{},entries=Object.entries(metrics);if(entries.length){const table=make('table','rollout-metrics'),body=make('tbody');for(const[key,row]of entries){const tr=make('tr'),value=row&&typeof row==='object'?row.value:row;const label=make('th','',row?.label||metricLabels[key]||key);label.scope='row';tr.append(label,make('td','',metricValue(value)),make('td','muted',row&&typeof row==='object'?[row.unit,labels[row.status]||row.status].filter(Boolean).join(' · '):''));body.append(tr);}table.append(body);result.append(table);}else result.append(make('p','muted','尚无完整定量结果。缺少运行或渲染证据时，不推断任务成功或安全能力。'));if(record.result)result.append(details('完整运行结果',record.result));target.append(result);if(record.task_critic_error){const failure=make('section','panel summary-panel');failure.append(make('h3','','Task critic 分析失败'),make('p','error',record.task_critic_error));target.append(failure);}if(record.task_critique&&(typeof record.task_critique==='string'||Object.keys(record.task_critique).length)){const critic=make('section','panel summary-panel');critic.append(make('h3','','Task critic 与共享经验'),make('p','text-block',typeof record.task_critique==='string'?record.task_critique:record.task_critique.summary||'已归档 task critic 分析。'));if(typeof record.task_critique==='object')critic.append(details('完整 critic 记录',record.task_critique));target.append(critic);}else if(!record.task_critic_error)target.append(make('p','muted small','此实验尚无 task critic 分析；可返回场景生成与 RSI 查看任务级记录。'));const evidence=make('div','rollout-evidence-grid');evidence.append(fieldPanel('评测 agent 设置',record.setup),fieldPanel('实验来源',record.scene_source));target.append(evidence);const execution=make('section','panel summary-panel');execution.id='rollout-execution';execution.append(make('h3','','评测 agent 执行过程'),make('p','muted small','正在读取调用与工具执行记录…'));target.append(execution);target.append(details('完整归档清单',record));}
async function refresh(){try{const data=await api('/api/rollouts');state.rows=Array.isArray(data)?data:data.rollouts||[];$('#connection').textContent='后端已连接';$('#connection-dot').classList.add('ok');renderList();if(state.id)await select(state.id,true);else{const row=state.rows.find(row=>!state.task||row.task_id===state.task);if(row)await select(row.id);else home();}}catch(error){$('#connection').textContent='连接中断';$('#connection-dot').classList.remove('ok');notice(error.message);}}
$('#refresh').onclick=refresh;$('#search').oninput=renderList;$('#menu').onclick=()=>document.body.classList.toggle('menu-open');document.addEventListener('click',event=>{if(document.body.classList.contains('menu-open')&&!event.target.closest('#sidebar')&&!event.target.closest('#menu'))document.body.classList.remove('menu-open');});document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh();});window.addEventListener('hashchange',()=>{const match=location.hash.match(/^#rollout\/(.+)$/);if(match)select(decodeURIComponent(match[1]));});const match=location.hash.match(/^#rollout\/(.+)$/);if(match)state.id=decodeURIComponent(match[1]);refresh();setInterval(()=>{if(!document.hidden)refresh();},5000);
