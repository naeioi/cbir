(function() {
    let $ = query => document.querySelector(query);
    let api = {
        "next_round": "cbic/next?choice=${choice}",
        "new_query":  "cbic/new",
		"image": "cbic/images/${image}" 
    }
    let left_img  = $("#left-wrapper > img");
    let right_img = $("#right-wrapper > img");
    
    let state = {
        img_urls: [],
        rounds: 0
    }
    
    function format(tmpl, pars) {
        let formatted = tmpl;
        for(let prop in pars)
            formatted = formatted.replace("${" + prop + "}", pars[prop])
        return formatted;
    }
    
    function reset() {
        state.rounds = 0;
        load_new_query();
    }
    
    function update() {
        left_img.src  = format(api["image"], { image: state.img_urls[0] });
        right_img.src = format(api["image"], { image: state.img_urls[1] });
        state.rounds++;
        $("#rounds").innerHTML = "" + state.rounds;
    }
    
    function load_new_query() {
        fetch(api["new_query"], { credentials: 'include' })
			.then(data => data.json())
			.then(imgs => {
				state.img_urls[0] = imgs[0];
				state.img_urls[1] = imgs[1];
				update();
			});
    }
    
    function load_next_round(choice) {
        fetch(format(api["next_round"], {choice}), { credentials: 'include' })
			.then(data => data.json())
			.then(imgs => {
				state.img_urls[0] = imgs[0];
				state.img_urls[1] = imgs[1];
				update();
			});
    }
    
    load_new_query();
	$("#left-wrapper  .closer").onclick = () => load_next_round(0);
	$("#right-wrapper .closer").onclick = () => load_next_round(1);
})()