// Camera geometry uses angles and scene-relative distances, independent of units.
export function normalizeCameraPose(value) {
  const result={};
  for(const name of ['position','target','up']) {
    const vector=value?.[name];
    if(!Array.isArray(vector)||vector.length!==3||vector.some(v=>typeof v!=='number'||!Number.isFinite(v)))throw Error(`camera.${name} must contain three finite numbers`);
    result[name]=vector.slice();
  }
  const direction=result.target.map((v,i)=>v-result.position[i]),distance=Math.hypot(...direction);
  if(!Number.isFinite(distance)||distance===0)throw Error('Camera position and target must differ');
  const forward=direction.map(v=>v/distance),length=Math.hypot(...result.up);
  let up=length?result.up.map(v=>v/length):[0,0,0];
  const project=axis=>{const dot=axis.reduce((sum,v,i)=>sum+v*forward[i],0);return axis.map((v,i)=>v-dot*forward[i]);};
  up=project(up);
  if(Math.hypot(...up)<1e-6){const index=forward.reduce((best,v,i)=>Math.abs(v)<Math.abs(forward[best])?i:best,0);up=project([0,1,2].map(i=>i===index?1:0));}
  const norm=Math.hypot(...up);result.up=up.map(v=>v/norm);
  return result;
}

export function cameraOrbitAxis(value) {
  const pose=normalizeCameraPose(value),length=Math.hypot(...value.up);
  if(!length||!Number.isFinite(length))return pose.up;
  const axis=value.up.map(v=>v/length),direction=pose.target.map((v,i)=>v-pose.position[i]),distance=Math.hypot(...direction),dot=axis.reduce((sum,v,i)=>sum+v*direction[i]/distance,0);
  return 1-Math.abs(dot)<1e-8?pose.up:axis;
}

export function cameraMinimumDistance(pose) {
  const distance=Math.hypot(...pose.target.map((v,i)=>v-pose.position[i]));
  const magnitude=Math.max(...pose.position.map(Math.abs),...pose.target.map(Math.abs));
  return Math.max(distance*1e-6,magnitude*Number.EPSILON*32,Number.MIN_VALUE);
}
