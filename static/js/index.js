window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function () {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function () {
        // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
        $(".navbar-burger").toggleClass("is-active");
        $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
        slidesToScroll: 1,
        slidesToShow: 3,
        loop: true,
        infinite: true,
        autoplay: false,
        autoplaySpeed: 3000,
    }

    // Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for (var i = 0; i < carousels.length; i++) {
        // Add listener to  event
        carousels[i].on('before:show', state => {
            console.log(state);
        });
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
        // bulmaCarousel instance is available as element.bulmaCarousel
        element.bulmaCarousel.on('before-show', function (state) {
            console.log(state);
        });
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/

})

// https://stackoverflow.com/a/47591788/5487412
$(document).ready(function () {
    $('#tabs li').on('click', function () {
        var tab = $(this).data('tab');

        $('#tabs li').removeClass('is-active');
        $(this).addClass('is-active');

        $('#tab-content div').removeClass('is-active');
        $('div[data-content="' + tab + '"]').addClass('is-active');
    });
});

// play video on mouseover
var mp4Gif = document.getElementsByClassName('mp4-gif');
for (var i = 0; i < mp4Gif.length; i++) {
    mp4Gif[i].addEventListener('mouseover', function () {
        this.play();
    }, false);
}

// method overview

var methodOverviewWrapper = document.getElementById('method-overview-wrapper');
methodOverviewWrapper.onclick = function (ev) {
    var target = ev.target;
    const rect = target.getBoundingClientRect();
    const x = ev.clientX - rect.left; //x position within the element.
    const y = ev.clientY - rect.top;  //y position within the element.
    var width = methodOverviewWrapper.offsetWidth;
    var height = methodOverviewWrapper.offsetHeight;
    var horizontal = (x < width / 2) ? "left" : "right";
    var vertical = (y < height / 2) ? "top" : "bottom";
    var id = "method-overview-" + vertical + "-" + horizontal;
    if (id == current) {
        return;
    }
    $(".method-overview").hide();
    $("." + id).show();
}

// get click outside method overview
$(document).mouseup(function (e) {
    var container = $("#method-overview-wrapper");
    if (!container.is(e.target) && container.has(e.target).length === 0) {
        $(".method-overview").hide();
        $(".method-overview-full-img").show()
    }
});

$(".method-overview-full-img").show()


// bibtex copy
// https://www.roboleary.net/2022/01/13/copy-code-to-clipboard-blog.html

const copyButtonLabel = "copy";

// use a class selector if available
let blocks = document.querySelectorAll("pre");

blocks.forEach((block) => {
    // only add button if browser supports Clipboard API
    if (navigator.clipboard) {
        let button = document.createElement("button");

        button.innerText = copyButtonLabel;
        block.appendChild(button);

        button.addEventListener("click", async () => {
            await copyCode(block, button);
        });
    }
});

async function copyCode(block, button) {
    let code = block.querySelector("code");
    let text = code.innerText;

    await navigator.clipboard.writeText(text);

    // visual feedback that task is completed
    button.innerText = "citation copied!";

    setTimeout(() => {
        button.innerText = copyButtonLabel;
    }, 700);
}
